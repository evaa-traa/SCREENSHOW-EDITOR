"""
AI Shorts Editor — Web Application
Flask-based web interface for YouTube Shorts video automation.
"""

import logging
import os
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, send_file, url_for
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ShortsEditor")

# Load env
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded .env from {env_path}")
    else:
        logger.warning(f".env not found at {env_path} — using system env vars")
except ImportError:
    logger.warning("python-dotenv not installed. Using system env vars.")

# Project dirs
PROJECT_DIR = str(Path(__file__).parent.resolve())
UPLOAD_DIR = os.path.join(PROJECT_DIR, "uploads")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
TEMP_DIR = os.path.join(PROJECT_DIR, "temp")

for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

# Check FFmpeg
ffmpeg_path = shutil.which("ffmpeg")
if ffmpeg_path:
    logger.info(f"FFmpeg found: {ffmpeg_path}")
else:
    logger.error("FFmpeg NOT found — video processing will fail.")

# Flask app
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload

# In-memory job tracking
# { job_id: { "status": "processing"|"done"|"error", "progress": 0-100, "message": "", "output_file": "" } }
jobs = {}


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle video upload and start processing."""
    try:
        # --- Validate file ---
        if "video" not in request.files:
            return jsonify({"error": "No video file uploaded."}), 400

        video_file = request.files["video"]
        if video_file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        # Check extension
        allowed_ext = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
        ext = os.path.splitext(video_file.filename)[1].lower()
        if ext not in allowed_ext:
            return jsonify({"error": f"Unsupported format: {ext}. Use: {', '.join(allowed_ext)}"}), 400

        # --- Save uploaded file ---
        job_id = str(uuid.uuid4())[:8]
        job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_upload_dir, exist_ok=True)

        safe_filename = f"input{ext}"
        input_path = os.path.join(job_upload_dir, safe_filename)
        video_file.save(input_path)

        file_size = os.path.getsize(input_path)
        if file_size == 0:
            shutil.rmtree(job_upload_dir, ignore_errors=True)
            return jsonify({"error": "Uploaded file is empty."}), 400

        logger.info(f"[{job_id}] Uploaded: {video_file.filename} ({file_size / 1024 / 1024:.1f}MB)")

        # --- Save music file if provided ---
        music_path = None
        if "music" in request.files and request.files["music"].filename:
            music_file = request.files["music"]
            music_ext = os.path.splitext(music_file.filename)[1].lower()
            music_path = os.path.join(job_upload_dir, f"music{music_ext}")
            music_file.save(music_path)

        # --- Save custom SRT if provided ---
        custom_srt_path = None
        if "srt" in request.files and request.files["srt"].filename:
            srt_file = request.files["srt"]
            custom_srt_path = os.path.join(job_upload_dir, "custom.srt")
            srt_file.save(custom_srt_path)

        # --- Parse options from form ---
        options = {
            "crop": request.form.get("crop", "true").lower() == "true",
            "crop_position": float(request.form.get("crop_position", "0.5")),
            "tint": request.form.get("tint", "false").lower() == "true",
            "tint_color": request.form.get("tint_color", "#FF8C00"),
            "tint_opacity": float(request.form.get("tint_opacity", "0.2")),
            "music": request.form.get("music_enabled", "false").lower() == "true",
            "music_path": music_path,
            "music_volume": float(request.form.get("music_volume", "0.2")),
            "watermark": request.form.get("watermark", "false").lower() == "true",
            "channel_name": request.form.get("channel_name", ""),
            "captions": request.form.get("captions", "false").lower() == "true",
            "caption_mode": request.form.get("caption_mode", "auto"),
            "caption_lang": request.form.get("caption_lang", "hi"),
            "srt_path": custom_srt_path,
        }

        # --- Build output path ---
        base_name = Path(video_file.filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_shorts_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, job_id, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # --- Initialize job tracking ---
        jobs[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting...",
            "output_file": output_path,
            "output_filename": output_filename,
            "upload_dir": job_upload_dir,
        }

        # --- Start processing in background thread ---
        thread = threading.Thread(
            target=_process_worker,
            args=(job_id, input_path, output_path, options),
            daemon=True,
        )
        thread.start()

        return jsonify({"job_id": job_id, "message": "Processing started."})

    except Exception as e:
        logger.error(f"Upload error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/status/<job_id>")
def status(job_id):
    """Poll processing status."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
    })


@app.route("/download/<job_id>")
def download(job_id):
    """Download the processed video."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    if job["status"] != "done":
        return jsonify({"error": "Video not ready yet."}), 400

    output_path = job["output_file"]
    if not os.path.isfile(output_path):
        return jsonify({"error": "Output file not found."}), 404

    return send_file(
        output_path,
        as_attachment=True,
        download_name=job["output_filename"],
        mimetype="video/mp4",
    )


@app.route("/cleanup/<job_id>", methods=["POST"])
def cleanup(job_id):
    """Clean up files for a completed job."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"ok": True})

    # Clean upload dir
    upload_dir = job.get("upload_dir", "")
    if upload_dir and os.path.isdir(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)

    # Clean output dir
    output_dir = os.path.dirname(job.get("output_file", ""))
    if output_dir and os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)

    jobs.pop(job_id, None)
    return jsonify({"ok": True})


# ─────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────

def _process_worker(job_id, input_path, output_path, options):
    """Run the video processing pipeline in a background thread."""
    from engine.processor import process_video, ProcessingError
    from engine.captions import generate_srt_groq

    job = jobs[job_id]

    def progress_callback(pct, msg):
        job["progress"] = round(pct, 1)
        job["message"] = msg

    try:
        # Handle auto-captions
        if options["captions"] and options["caption_mode"] == "auto":
            progress_callback(2, "Generating captions via Groq Whisper...")
            api_key = os.environ.get("GROQ_API_KEY", "")
            srt_output = os.path.join(TEMP_DIR, f"{job_id}_captions.srt")
            try:
                generate_srt_groq(
                    input_path, srt_output, api_key,
                    language=options.get("caption_lang", "hi"),
                )
                options["srt_path"] = srt_output
                progress_callback(10, "Captions generated!")
            except Exception as e:
                logger.warning(f"[{job_id}] Caption generation failed: {e}")
                progress_callback(10, f"Captions skipped: {e}")
                options["captions"] = False

        # Process video
        process_video(
            input_path=input_path,
            output_path=output_path,
            options=options,
            progress_callback=progress_callback,
            temp_dir=os.path.join(TEMP_DIR, job_id),
        )

        job["status"] = "done"
        job["progress"] = 100
        job["message"] = "Complete!"
        logger.info(f"[{job_id}] Processing complete: {output_path}")

    except ProcessingError as e:
        job["status"] = "error"
        job["message"] = str(e)
        logger.error(f"[{job_id}] Processing failed: {e}")
    except Exception as e:
        job["status"] = "error"
        job["message"] = f"Unexpected error: {e}"
        logger.error(f"[{job_id}] Unexpected error: {e}", exc_info=True)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
