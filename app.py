"""
AI Shorts Editor — Web Application
Flask-based web interface for YouTube Shorts video automation.
"""

import logging
import os
import shutil
import sys
import threading
import uuid
import json
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file

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
JOB_STATE_DIR = os.path.join(TEMP_DIR, "job_state")

for d in [UPLOAD_DIR, OUTPUT_DIR, TEMP_DIR, JOB_STATE_DIR]:
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


def _job_state_path(job_id: str) -> str:
    return os.path.join(JOB_STATE_DIR, f"{job_id}.json")


def _save_job(job_id: str, payload: dict):
    temp_path = _job_state_path(job_id) + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)
    os.replace(temp_path, _job_state_path(job_id))


def _load_job(job_id: str):
    path = _job_state_path(job_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        logger.warning("Failed to read job state for %s", job_id, exc_info=True)
        return None


def _update_job(job_id: str, **changes):
    job = _load_job(job_id)
    if job is None:
        return
    job.update(changes)
    _save_job(job_id, job)


def _delete_job(job_id: str):
    path = _job_state_path(job_id)
    if os.path.isfile(path):
        os.remove(path)

# In-memory job tracking
# { job_id: { "status": "processing"|"done"|"error", "progress": 0-100, "message": "", "output_file": "" } }


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
        uploaded_videos = request.files.getlist("videos")
        if not uploaded_videos:
            single_video = request.files.get("video")
            if single_video and single_video.filename:
                uploaded_videos = [single_video]

        if not uploaded_videos:
            return jsonify({"error": "No video file uploaded."}), 400

        # Check extension
        allowed_ext = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}

        # --- Save uploaded file ---
        job_id = str(uuid.uuid4())[:8]
        job_upload_dir = os.path.join(UPLOAD_DIR, job_id)
        os.makedirs(job_upload_dir, exist_ok=True)

        source_videos = []
        total_size = 0
        primary_name = None
        for index, video_file in enumerate(uploaded_videos):
            if not video_file or video_file.filename == "":
                continue
            ext = os.path.splitext(video_file.filename)[1].lower()
            if ext not in allowed_ext:
                shutil.rmtree(job_upload_dir, ignore_errors=True)
                return jsonify({"error": f"Unsupported format: {ext}. Use: {', '.join(allowed_ext)}"}), 400

            safe_filename = f"source_{index}{ext}"
            input_path = os.path.join(job_upload_dir, safe_filename)
            video_file.save(input_path)

            file_size = os.path.getsize(input_path)
            if file_size == 0:
                shutil.rmtree(job_upload_dir, ignore_errors=True)
                return jsonify({"error": f"Uploaded file is empty: {video_file.filename}"}), 400

            total_size += file_size
            if primary_name is None:
                primary_name = video_file.filename
            source_videos.append(
                {
                    "path": input_path,
                    "original_name": video_file.filename,
                }
            )

        if not source_videos:
            shutil.rmtree(job_upload_dir, ignore_errors=True)
            return jsonify({"error": "No usable video files were uploaded."}), 400

        logger.info(
            "[%s] Uploaded %d video(s) (%.1fMB total)",
            job_id,
            len(source_videos),
            total_size / 1024 / 1024,
        )

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
        caption_box = _parse_caption_box(request.form.get("caption_box", ""))
        options = {
            "crop": _parse_bool(request.form.get("crop"), True),
            "crop_position": _parse_float(request.form.get("crop_position"), 0.5, 0.0, 1.0),
            "source_rotation": request.form.get("source_rotation", "none"),
            "source_fit_mode": request.form.get("source_fit_mode", "cover"),
            "source_pan_x": _parse_float(request.form.get("source_pan_x"), 0.0, -1.0, 1.0),
            "source_pan_y": _parse_float(request.form.get("source_pan_y"), 0.0, -1.0, 1.0),
            "source_zoom": _parse_float(request.form.get("source_zoom"), 1.0, 0.6, 2.5),
            "look_preset": request.form.get("look_preset", "warm_cinematic"),
            "look_strength": _parse_float(request.form.get("look_strength"), 0.85, 0.0, 1.0),
            "look_motion": _parse_float(request.form.get("look_motion"), 0.45, 0.0, 1.0),
            "text_mode": request.form.get("text_mode", "none"),
            "text_primary": request.form.get("text_primary", ""),
            "text_secondary": request.form.get("text_secondary", ""),
            "text_highlight": request.form.get("text_highlight", ""),
            "text_accent_color": request.form.get("text_accent_color", "#18D7FF"),
            "highlight_color": request.form.get("highlight_color", "#FF7B47"),
            "text_bold": _parse_bool(request.form.get("text_bold"), True),
            "text_scale": _parse_float(request.form.get("text_scale"), 1.0, 0.7, 1.4),
            "top_text_scale": _parse_float(request.form.get("top_text_scale"), 1.0, 0.7, 2.0),
            "text_box": _parse_text_box(request.form.get("text_box", "")),
            "trim_silence": _parse_bool(request.form.get("trim_silence"), False),
            "silence_threshold_db": _parse_float(request.form.get("silence_threshold_db"), -45.0, -80.0, -5.0),
            "min_silence_duration": _parse_float(request.form.get("min_silence_duration"), 0.35, 0.05, 3.0),
            "silence_padding": _parse_float(request.form.get("silence_padding"), 0.1, 0.0, 1.0),
            "tint": _parse_bool(request.form.get("tint"), False),
            "tint_color": request.form.get("tint_color", "#FF8C00"),
            "tint_opacity": _parse_float(request.form.get("tint_opacity"), 0.2, 0.0, 0.6),
            "music": _parse_bool(request.form.get("music_enabled"), False),
            "music_path": music_path,
            "music_volume": _parse_float(request.form.get("music_volume"), 0.2, 0.0, 1.0),
            "duck_music": _parse_bool(request.form.get("duck_music"), True),
            "ducking_strength": _parse_float(request.form.get("ducking_strength"), 0.7, 0.0, 1.0),
            "audio_boost": _parse_float(request.form.get("audio_boost"), 1.0, 1.0, 2.5),
            "watermark": _parse_bool(request.form.get("watermark"), False),
            "channel_name": request.form.get("channel_name", ""),
            "channel_position": request.form.get("channel_position", "lower_left_overlay"),
            "captions": _parse_bool(request.form.get("captions"), False),
            "caption_mode": request.form.get("caption_mode", "auto"),
            "caption_lang": request.form.get("caption_lang", "auto"),
            "caption_style": request.form.get("caption_style", "reels"),
            "caption_font_size": _parse_int(request.form.get("caption_font_size"), 72, 28, 160),
            "caption_max_words": _parse_int(request.form.get("caption_max_words"), 4, 1, 8),
            "caption_box": caption_box,
            "srt_path": custom_srt_path,
            "caption_path": custom_srt_path,
            "caption_format": "srt" if custom_srt_path else "",
            "export_quality": request.form.get("export_quality", "high"),
            "source_videos": source_videos,
            "segments": _parse_segments(request.form.get("segments_json", "")),
        }

        # --- Build output path ---
        base_name = Path(primary_name or "clip").stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_shorts_{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, job_id, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # --- Initialize job tracking ---
        job_payload = {
            "status": "processing",
            "progress": 0,
            "message": "Starting...",
            "output_file": output_path,
            "output_filename": output_filename,
            "upload_dir": job_upload_dir,
        }
        _save_job(job_id, job_payload)

        # --- Start processing in background thread ---
        thread = threading.Thread(
            target=_process_worker,
            args=(job_id, source_videos[0]["path"], output_path, options),
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
    job = _load_job(job_id)
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
    job = _load_job(job_id)
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
    job = _load_job(job_id)
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

    _delete_job(job_id)
    return jsonify({"ok": True})


# ─────────────────────────────────────────────
# Background worker
# ─────────────────────────────────────────────

def _process_worker(job_id, input_path, output_path, options):
    """Run the video processing pipeline in a background thread."""
    from engine.processor import build_raw_clip, process_video, ProcessingError

    def progress_callback(pct, msg):
        _update_job(job_id, progress=round(pct, 1), message=msg)

    try:
        working_input = input_path
        source_videos = options.get("source_videos") or [{"path": input_path}]
        segments = options.get("segments") or []
        source_rotation = str(options.get("source_rotation", "none") or "none").strip().lower()
        source_fit_mode = str(options.get("source_fit_mode", "cover") or "cover").strip().lower()
        source_pan_x = _parse_float(options.get("source_pan_x"), 0.0, -1.0, 1.0)
        source_pan_y = _parse_float(options.get("source_pan_y"), 0.0, -1.0, 1.0)
        source_zoom = _parse_float(options.get("source_zoom"), 1.0, 0.6, 2.5)
        trim_silence = _parse_bool(options.get("trim_silence"), False)
        has_manual_source_edits = (
            source_rotation != "none"
            or source_fit_mode != "cover"
            or abs(source_pan_x) > 0.001
            or abs(source_pan_y) > 0.001
            or abs(source_zoom - 1.0) > 0.001
        )
        options["input_start"] = None
        options["input_end"] = None

        if len(source_videos) == 1 and not segments and not has_manual_source_edits and not trim_silence:
            working_input = source_videos[0]["path"]
            options["source_prepared"] = False
        elif source_videos:
            raw_clip_path = os.path.join(TEMP_DIR, f"{job_id}_raw.mp4")
            progress_callback(1, "Building raw clip...")
            build_raw_clip(
                source_videos=source_videos,
                segments=segments,
                output_path=raw_clip_path,
                crop_position=options.get("crop_position", 0.5),
                source_fit_mode=options.get("source_fit_mode", "cover"),
                source_rotation=options.get("source_rotation", "none"),
                source_pan_x=options.get("source_pan_x", 0.0),
                source_pan_y=options.get("source_pan_y", 0.0),
                source_zoom=options.get("source_zoom", 1.0),
                trim_silence=options.get("trim_silence", False),
                silence_threshold_db=options.get("silence_threshold_db", -45.0),
                min_silence_duration=options.get("min_silence_duration", 0.35),
                silence_padding=options.get("silence_padding", 0.1),
                progress_callback=progress_callback,
                progress_range=(1, 22),
            )
            working_input = raw_clip_path
            options["source_prepared"] = True
            options["source_rotation"] = "none"
            options["source_fit_mode"] = "cover"
            options["source_pan_x"] = 0.0
            options["source_pan_y"] = 0.0
            options["source_zoom"] = 1.0

        caption_settings = {
            "preset": options.get("caption_style", "reels"),
            "font_size": options.get("caption_font_size", 72),
            "max_words": options.get("caption_max_words", 4),
            "box": options.get("caption_box"),
        }

        # Handle captions
        if options["captions"]:
            from engine.captions import convert_srt_to_ass, generate_ass_groq

            ass_output = os.path.join(TEMP_DIR, f"{job_id}_captions.ass")
            if options["caption_mode"] == "auto":
                progress_callback(2, "Generating animated captions...")
                api_key = os.environ.get("GROQ_API_KEY", "")
                try:
                    generate_ass_groq(
                        working_input,
                        ass_output,
                        api_key,
                        language=options.get("caption_lang", "auto"),
                        settings=caption_settings,
                    )
                    options["caption_path"] = ass_output
                    options["caption_format"] = "ass"
                    progress_callback(10, "Animated captions ready.")
                except Exception as e:
                    logger.warning(f"[{job_id}] Caption generation failed: {e}")
                    progress_callback(10, f"Captions skipped: {e}")
                    options["captions"] = False
            elif options.get("srt_path"):
                progress_callback(2, "Styling uploaded captions...")
                try:
                    convert_srt_to_ass(
                        options["srt_path"],
                        ass_output,
                        settings=caption_settings,
                    )
                    options["caption_path"] = ass_output
                    options["caption_format"] = "ass"
                    progress_callback(10, "Custom captions styled.")
                except Exception as e:
                    logger.warning(f"[{job_id}] Caption styling failed: {e}")
                    progress_callback(10, f"Captions skipped: {e}")
                    options["captions"] = False

        # Process video
        process_video(
            input_path=working_input,
            output_path=output_path,
            options=options,
            progress_callback=progress_callback,
            temp_dir=os.path.join(TEMP_DIR, job_id),
        )

        _update_job(job_id, status="done", progress=100, message="Complete!")
        logger.info(f"[{job_id}] Processing complete: {output_path}")

    except ProcessingError as e:
        _update_job(job_id, status="error", message=str(e))
        logger.error(f"[{job_id}] Processing failed: {e}")
    except Exception as e:
        _update_job(job_id, status="error", message=f"Unexpected error: {e}")
        logger.error(f"[{job_id}] Unexpected error: {e}", exc_info=True)


def _parse_caption_box(raw_value: str) -> dict:
    """Parse a normalized caption box from JSON form data."""
    fallback = {"x": 0.08, "y": 0.72, "w": 0.84, "h": 0.18}
    if not raw_value:
        return fallback

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    try:
        x = max(0.0, min(0.9, float(parsed.get("x", fallback["x"]))))
        y = max(0.0, min(0.94, float(parsed.get("y", fallback["y"]))))
        w = max(0.12, min(1.0 - x, float(parsed.get("w", fallback["w"]))))
        h = max(0.08, min(1.0 - y, float(parsed.get("h", fallback["h"]))))
    except (TypeError, ValueError):
        return fallback

    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


def _parse_text_box(raw_value: str) -> dict:
    """Parse a normalized text placement box from JSON form data."""
    fallback = {"x": 0.14, "y": 0.38, "w": 0.72, "h": 0.2}
    if not raw_value:
        return fallback

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return fallback

    if not isinstance(parsed, dict):
        return fallback

    try:
        x = max(0.0, min(0.88, float(parsed.get("x", fallback["x"]))))
        y = max(0.0, min(0.94, float(parsed.get("y", fallback["y"]))))
        w = max(0.12, min(1.0 - x, float(parsed.get("w", fallback["w"]))))
        h = max(0.08, min(1.0 - y, float(parsed.get("h", fallback["h"]))))
    except (TypeError, ValueError):
        return fallback

    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


def _parse_segments(raw_value: str) -> list:
    """Parse segment timeline definitions from JSON form data."""
    if not raw_value:
        return []

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        return []

    if not isinstance(parsed, list):
        return []

    segments = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        start_value = _parse_timecode(item.get("start", 0))
        end_value = _parse_timecode(item.get("end", 0))
        if start_value is None or end_value is None:
            continue
        segments.append(
            {
                "video_index": item.get("video_index", 0),
                "start": start_value,
                "end": end_value,
            }
        )
    return segments


def _parse_timecode(raw_value):
    """Parse seconds or HH:MM:SS / MM:SS strings into float seconds."""
    if raw_value is None:
        return None

    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    text = str(raw_value).strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    if len(parts) not in {2, 3}:
        return None

    try:
        parts = [float(part) for part in parts]
    except ValueError:
        return None

    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds

    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def _parse_bool(raw_value, default: bool) -> bool:
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(raw_value, default: float, low: float = None, high: float = None) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _parse_int(raw_value, default: int, low: int = None, high: int = None) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
