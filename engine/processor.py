"""
Video Processing Engine — FFmpeg-based pipeline for Shorts editing.

Uses direct subprocess calls for maximum control and error visibility.
Every FFmpeg operation captures stderr, checks return codes, and reports
meaningful errors to the caller.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("ShortsEditor.Processor")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class ProcessingError(Exception):
    """Raised when any step in the video pipeline fails."""
    pass


class FFmpegNotFoundError(ProcessingError):
    """Raised when FFmpeg/FFprobe is not available on the system."""
    pass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _find_ffmpeg():
    """Return the path to ffmpeg, or raise if not found."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError(
            "FFmpeg is not installed or not in PATH.\n"
            "Download from https://ffmpeg.org/download.html and add to PATH."
        )
    return path


def _find_ffprobe():
    """Return the path to ffprobe, or raise if not found."""
    path = shutil.which("ffprobe")
    if path is None:
        raise FFmpegNotFoundError(
            "FFprobe is not installed or not in PATH.\n"
            "It usually comes bundled with FFmpeg."
        )
    return path


def _run_ffmpeg(args: list, description: str, duration: float = None,
                progress_callback=None, progress_range: tuple = None):
    """
    Run an FFmpeg command with full error capture.

    Parameters
    ----------
    args : list
        Full command list (including 'ffmpeg' as first element).
    description : str
        Human-readable name of this step (for error messages).
    duration : float, optional
        Total duration in seconds (for progress calculation).
    progress_callback : callable, optional
        Function(percent: float, status: str) to report progress.
    progress_range : tuple, optional
        (start_pct, end_pct) — the portion of overall progress this step covers.

    Raises
    ------
    ProcessingError
        If FFmpeg returns a non-zero exit code.
    """
    logger.info(f"[{description}] Running: {' '.join(args)}")

    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    stderr_lines = []
    start_pct = progress_range[0] if progress_range else 0
    end_pct = progress_range[1] if progress_range else 100

    # Read stderr line-by-line for progress parsing
    for line in process.stderr:
        stderr_lines.append(line)
        # Parse progress from FFmpeg output: "time=00:01:23.45"
        if duration and progress_callback and "time=" in line:
            match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
            if match:
                h, m, s = float(match.group(1)), float(match.group(2)), float(match.group(3))
                current_time = h * 3600 + m * 60 + s
                step_progress = min(current_time / duration, 1.0)
                overall_pct = start_pct + step_progress * (end_pct - start_pct)
                progress_callback(overall_pct, description)

    process.wait()

    if process.returncode != 0:
        stderr_text = "".join(stderr_lines[-30:])  # Last 30 lines for context
        logger.error(f"[{description}] FFmpeg failed (code {process.returncode}):\n{stderr_text}")
        raise ProcessingError(
            f"{description} failed.\n\n"
            f"FFmpeg exit code: {process.returncode}\n"
            f"Error output:\n{stderr_text}"
        )

    logger.info(f"[{description}] Completed successfully.")


def probe_video(input_path: str) -> dict:
    """
    Use ffprobe to extract video metadata.

    Returns
    -------
    dict with keys: width, height, duration, has_audio
    """
    ffprobe = _find_ffprobe()
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        input_path
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        raise ProcessingError(f"FFprobe timed out reading: {input_path}")

    if result.returncode != 0:
        raise ProcessingError(
            f"Cannot read video file.\n"
            f"FFprobe error: {result.stderr[:500]}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise ProcessingError("FFprobe returned invalid data. File may be corrupted.")

    # Find video stream
    video_stream = None
    has_audio = False
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        if stream.get("codec_type") == "audio":
            has_audio = True

    if video_stream is None:
        raise ProcessingError("No video stream found in the file.")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    if width == 0 or height == 0:
        raise ProcessingError("Could not determine video dimensions.")

    # Get duration (try stream, then format)
    duration = 0.0
    if "duration" in video_stream:
        duration = float(video_stream["duration"])
    elif "duration" in data.get("format", {}):
        duration = float(data["format"]["duration"])

    if duration <= 0:
        raise ProcessingError("Could not determine video duration. File may be invalid.")

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "has_audio": has_audio,
    }


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------

def process_video(
    input_path: str,
    output_path: str,
    options: dict,
    progress_callback=None,
    temp_dir: str = None,
):
    """
    Main processing pipeline. Orchestrates all editing steps.

    Parameters
    ----------
    input_path : str
        Path to the source video file.
    output_path : str
        Path for the final exported MP4.
    options : dict
        {
            "crop": bool,               # Crop to 9:16
            "crop_position": float,     # 0.0 (top) to 1.0 (bottom), default 0.5 (center)
            "captions": bool,           # Burn subtitles
            "srt_path": str or None,    # Path to .srt file (auto-generated or custom)
            "music": bool,              # Add background music
            "music_path": str or None,  # Path to music file
            "music_volume": float,      # 0.0 to 1.0, default 0.2
            "tint": bool,              # Apply color tint
            "tint_color": str,         # Hex color e.g. "#FF0000"
            "tint_opacity": float,     # 0.0 to 1.0, default 0.2
            "watermark": bool,         # Add channel name
            "channel_name": str,       # Text to display
        }
    progress_callback : callable, optional
        Function(percent: float, status: str).
    temp_dir : str, optional
        Directory for temp files. Created if needed, cleaned on completion.
    """

    ffmpeg = _find_ffmpeg()

    # --- Validate input ---
    if not os.path.isfile(input_path):
        raise ProcessingError(f"Input file not found: {input_path}")

    file_size = os.path.getsize(input_path)
    if file_size == 0:
        raise ProcessingError("Input file is empty (0 bytes).")

    if progress_callback:
        progress_callback(1, "Analyzing video...")

    info = probe_video(input_path)
    logger.info(f"Video info: {info}")

    # --- Setup temp dir ---
    own_temp = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="shorts_editor_")
        own_temp = True
    else:
        os.makedirs(temp_dir, exist_ok=True)

    try:
        _run_pipeline(
            ffmpeg, input_path, output_path, options, info,
            temp_dir, progress_callback
        )
    finally:
        # Always clean temp files
        if own_temp:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Cleaned temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean temp dir: {e}")

    # --- Verify output ---
    if not os.path.isfile(output_path):
        raise ProcessingError("Processing completed but output file was not created.")

    out_size = os.path.getsize(output_path)
    if out_size == 0:
        os.remove(output_path)
        raise ProcessingError("Processing completed but output file is empty.")

    if progress_callback:
        progress_callback(100, "Done!")

    logger.info(f"Processing complete. Output: {output_path} ({out_size / 1024 / 1024:.1f} MB)")


def _run_pipeline(ffmpeg, input_path, output_path, options, info,
                  temp_dir, progress_callback):
    """Build and execute the FFmpeg filter chain."""

    src_w = info["width"]
    src_h = info["height"]
    duration = info["duration"]
    has_audio = info["has_audio"]

    crop_enabled = options.get("crop", True)
    crop_pos = options.get("crop_position", 0.5)  # 0=top, 0.5=center, 1=bottom
    tint_enabled = options.get("tint", False)
    tint_color = options.get("tint_color", "#000000")
    tint_opacity = options.get("tint_opacity", 0.2)
    watermark_enabled = options.get("watermark", False)
    channel_name = options.get("channel_name", "")
    captions_enabled = options.get("captions", False)
    srt_path = options.get("srt_path", None)
    music_enabled = options.get("music", False)
    music_path = options.get("music_path", None)
    music_volume = options.get("music_volume", 0.2)

    # ---- Build video filter chain ----
    vfilters = []

    # Step 1: Crop to 9:16
    if crop_enabled:
        target_ratio = 9 / 16
        src_ratio = src_w / src_h

        if src_ratio > target_ratio:
            # Video is wider than 9:16 — crop horizontally
            crop_h = src_h
            crop_w = int(src_h * target_ratio)
            # Center horizontally (crop_position not relevant for horizontal crop)
            x_offset = (src_w - crop_w) // 2
            y_offset = 0
            vfilters.append(f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset}")
        elif src_ratio < target_ratio:
            # Video is taller than 9:16 — crop vertically
            crop_w = src_w
            crop_h = int(src_w / target_ratio)
            # Use crop_position to determine vertical offset
            max_offset = src_h - crop_h
            y_offset = int(max_offset * crop_pos)
            x_offset = 0
            vfilters.append(f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset}")
        # else: already 9:16, no crop needed

    # Step 2: Resize to 1080x1920
    vfilters.append("scale=1080:1920:flags=lanczos")
    # Ensure even dimensions (required by most codecs)
    vfilters.append("setsar=1")

    # Step 3: Tint overlay
    if tint_enabled and tint_color:
        # Convert hex to RGB
        hex_clean = tint_color.lstrip("#")
        try:
            r = int(hex_clean[0:2], 16)
            g = int(hex_clean[2:4], 16)
            b = int(hex_clean[4:6], 16)
        except (ValueError, IndexError):
            logger.warning(f"Invalid tint color '{tint_color}', skipping tint.")
            r, g, b = 0, 0, 0
            tint_enabled = False

        if tint_enabled:
            opacity = max(0.0, min(1.0, tint_opacity))
            # Use colorchannelmixer for a tint effect
            # drawbox overlay approach is simpler and more reliable
            vfilters.append(
                f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{hex_clean}@{opacity}:t=fill"
            )

    # Step 4: Watermark (channel name)
    if watermark_enabled and channel_name.strip():
        safe_name = channel_name.strip().replace("'", "'\\''").replace(":", r"\:")
        # Bottom center, semi-transparent white text
        vfilters.append(
            f"drawtext=text='{safe_name}'"
            f":fontsize=28"
            f":fontcolor=white@0.6"
            f":x=(w-text_w)/2"
            f":y=h-th-60"
            f":borderw=1"
            f":bordercolor=black@0.4"
        )

    # Step 5: Captions (subtitles)
    if captions_enabled and srt_path and os.path.isfile(srt_path):
        # FFmpeg subtitles filter needs forward slashes and escaped colons/backslashes
        srt_escaped = srt_path.replace("\\", "/").replace(":", r"\:")
        vfilters.append(
            f"subtitles='{srt_escaped}'"
            f":force_style='FontSize=22,FontName=Arial,PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,Outline=2,Bold=1,"
            f"Alignment=2,MarginV=80'"
        )
    elif captions_enabled and (srt_path is None or not os.path.isfile(srt_path or "")):
        logger.warning("Captions enabled but no SRT file found. Skipping captions.")

    # ---- Build audio filter chain ----
    # We need to handle: original audio + optional background music
    audio_inputs = []
    audio_filters = []
    input_args = ["-i", input_path]
    input_count = 1

    if music_enabled and music_path and os.path.isfile(music_path):
        # Add music as second input, loop it
        input_args.extend(["-stream_loop", "-1", "-i", music_path])
        music_idx = input_count
        input_count += 1

        vol = max(0.0, min(1.0, music_volume))

        if has_audio:
            # Mix original audio + music
            audio_filters.append(
                f"[{music_idx}:a]volume={vol},atrim=0:{duration},asetpts=PTS-STARTPTS[bgm];"
                f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
        else:
            # Only music (no original audio)
            audio_filters.append(
                f"[{music_idx}:a]volume={vol},atrim=0:{duration},asetpts=PTS-STARTPTS[aout]"
            )
    elif has_audio:
        # Just pass through original audio
        audio_filters.append("[0:a]acopy[aout]")
    # else: no audio at all

    # ---- Combine into final FFmpeg command ----
    vfilter_str = ",".join(vfilters)

    # Build complex filter graph
    filter_parts = []
    filter_parts.append(f"[0:v]{vfilter_str}[vout]")
    if audio_filters:
        filter_parts.extend(audio_filters)

    filter_graph = ";".join(filter_parts)

    cmd = [ffmpeg, "-y"]  # Overwrite output
    cmd.extend(input_args)
    cmd.extend(["-filter_complex", filter_graph])
    cmd.extend(["-map", "[vout]"])

    if audio_filters:
        cmd.extend(["-map", "[aout]"])

    # Output settings
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-r", "30",
        "-pix_fmt", "yuv420p",
    ])

    if audio_filters:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])

    cmd.extend([
        "-movflags", "+faststart",
        "-t", str(duration),  # Ensure output matches source duration
        output_path,
    ])

    if progress_callback:
        progress_callback(5, "Processing video...")

    _run_ffmpeg(
        cmd,
        description="Video processing",
        duration=duration,
        progress_callback=progress_callback,
        progress_range=(5, 95),
    )

    if progress_callback:
        progress_callback(95, "Finalizing...")


def extract_audio(input_path: str, output_wav_path: str):
    """
    Extract audio from a video file as a WAV for transcription.

    Parameters
    ----------
    input_path : str
        Path to the video file.
    output_wav_path : str
        Path where the WAV file will be saved.
    """
    ffmpeg = _find_ffmpeg()

    cmd = [
        ffmpeg, "-y",
        "-i", input_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",  # WAV format
        "-ar", "16000",          # 16kHz (Whisper optimal)
        "-ac", "1",              # Mono
        output_wav_path,
    ]

    _run_ffmpeg(cmd, description="Extracting audio")
