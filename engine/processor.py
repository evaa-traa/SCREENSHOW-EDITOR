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
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("ShortsEditor.Processor")

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_EMOJI_FONT = PROJECT_ROOT / "assets" / "fonts" / "NotoColorEmoji_WindowsCompatible.ttf"

LOOK_PRESETS = {
    "warm_cinematic": {
        "eq": {"contrast": 1.12, "saturation": 1.16, "brightness": 0.015, "gamma": 1.03},
        "pulse": {"contrast": 0.03, "saturation": 0.08, "gamma": 0.025},
        "colorbalance": {"rs": 0.10, "gs": 0.02, "bs": -0.07, "rm": 0.05, "bm": -0.02},
        "vignette": 0.22,
        "sharpen": 0.75,
    },
    "cool_teal": {
        "eq": {"contrast": 1.10, "saturation": 1.10, "brightness": 0.008, "gamma": 1.01},
        "pulse": {"contrast": 0.025, "saturation": 0.06, "gamma": 0.018},
        "colorbalance": {"rs": -0.04, "gs": 0.03, "bs": 0.10, "gm": 0.02, "bm": 0.04},
        "vignette": 0.18,
        "sharpen": 0.65,
    },
    "muted_drama": {
        "eq": {"contrast": 1.15, "saturation": 0.88, "brightness": -0.005, "gamma": 1.04},
        "pulse": {"contrast": 0.02, "saturation": 0.04, "gamma": 0.02},
        "colorbalance": {"rs": 0.04, "gs": 0.02, "bs": -0.05, "rm": 0.03, "bm": -0.03},
        "vignette": 0.26,
        "sharpen": 0.7,
    },
    "black_white": {
        "eq": {"contrast": 1.18, "saturation": 0.0, "brightness": 0.01, "gamma": 1.05},
        "pulse": {"contrast": 0.025, "saturation": 0.0, "gamma": 0.015},
        "colorbalance": {},
        "vignette": 0.28,
        "sharpen": 0.8,
    },
}

DEFAULT_DRAW_FONT_FAMILIES = "DejaVu Sans,Noto Sans,Arial,Helvetica"
EXTENDED_DRAW_FONT_FAMILIES = "Noto Sans,DejaVu Sans,Noto Emoji,Noto Color Emoji,Segoe UI Emoji,Apple Color Emoji,Symbola"


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
            "source_rotation": str,     # none|cw|ccw|180
            "source_fit_mode": str,     # cover|contain
            "source_pan_x": float,      # -1.0 to 1.0 manual horizontal framing
            "source_pan_y": float,      # -1.0 to 1.0 manual vertical framing
            "source_zoom": float,       # 0.6 to 2.5 manual zoom
            "look_preset": str,         # Cinematic grading preset
            "look_strength": float,     # 0.0 to 1.0
            "look_motion": float,       # 0.0 to 1.0 subtle animated mood shift
            "text_mode": str,           # none|center_title|premium_subtitle|top_commentary
            "text_primary": str,        # Main template text
            "text_secondary": str,      # Optional second line
            "text_accent_color": str,   # Hex color for highlighted text
            "text_scale": float,        # 0.7 to 1.4
            "text_box": dict,           # Normalized x/y/w/h placement box
            "captions": bool,           # Burn subtitles
            "caption_path": str or None, # Path to .ass or .srt subtitles
            "caption_format": str,      # "ass" or "srt"
            "music": bool,              # Add background music
            "music_path": str or None,  # Path to music file
            "music_volume": float,      # 0.0 to 1.0, default 0.2
            "tint": bool,              # Apply color tint
            "tint_color": str,         # Hex color e.g. "#FF0000"
            "tint_opacity": float,     # 0.0 to 1.0, default 0.2
            "watermark": bool,         # Add channel name
            "channel_name": str,       # Text to display
            "export_quality": str,     # "high", "balanced", or "fast"
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
    source_rotation = str(options.get("source_rotation", "none") or "none").strip().lower()
    source_fit_mode = str(options.get("source_fit_mode", "cover") or "cover").strip().lower()
    source_pan_x = _clamp_float(options.get("source_pan_x", 0.0), -1.0, 1.0)
    source_pan_y = _clamp_float(options.get("source_pan_y", 0.0), -1.0, 1.0)
    source_zoom = _clamp_float(options.get("source_zoom", 1.0), 0.6, 2.5)
    source_prepared = bool(options.get("source_prepared", False))
    input_start = options.get("input_start")
    input_end = options.get("input_end")
    look_preset = str(options.get("look_preset", "warm_cinematic")).strip().lower()
    look_strength = _clamp_float(options.get("look_strength", 0.85), 0.0, 1.0)
    look_motion = _clamp_float(options.get("look_motion", 0.45), 0.0, 1.0)
    text_mode = str(options.get("text_mode", "none")).strip().lower()
    text_primary = str(options.get("text_primary", "") or "")
    text_secondary = str(options.get("text_secondary", "") or "")
    text_highlight = str(options.get("text_highlight", "") or "")
    text_accent_color = str(options.get("text_accent_color", "#18D7FF") or "#18D7FF")
    highlight_color = str(options.get("highlight_color", "#FF7B47") or "#FF7B47")
    text_bold = bool(options.get("text_bold", True))
    text_scale = _clamp_float(options.get("text_scale", 1.0), 0.7, 1.4)
    top_text_scale = _clamp_float(options.get("top_text_scale", 1.0), 0.7, 2.0)
    text_box = _normalize_text_box(options.get("text_box"))
    tint_enabled = options.get("tint", False)
    tint_color = options.get("tint_color", "#000000")
    tint_opacity = options.get("tint_opacity", 0.2)
    watermark_enabled = options.get("watermark", False)
    channel_name = options.get("channel_name", "")
    channel_position = str(options.get("channel_position", "lower_left_overlay") or "lower_left_overlay").strip().lower()
    captions_enabled = options.get("captions", False)
    caption_path = options.get("caption_path") or options.get("srt_path")
    caption_format = options.get("caption_format", "")
    music_enabled = options.get("music", False)
    music_path = options.get("music_path", None)
    music_volume = options.get("music_volume", 0.2)
    duck_music = bool(options.get("duck_music", True))
    ducking_strength = _clamp_float(options.get("ducking_strength", 0.7), 0.0, 1.0)
    audio_boost = _clamp_float(options.get("audio_boost", 1.0), 1.0, 2.5)
    export_quality = str(options.get("export_quality", "high")).strip().lower()

    if input_start is not None:
        input_start = _clamp_float(input_start, 0.0, duration)
    if input_end is not None:
        input_end = _clamp_float(input_end, 0.0, duration)
    if input_start is not None or input_end is not None:
        trim_start = input_start or 0.0
        trim_end = input_end if input_end is not None else duration
        if trim_end <= trim_start:
            raise ProcessingError("Selected trim range is invalid.")
        duration = trim_end - trim_start

    # ---- Build video filter chain ----
    vfilters = []

    if not source_prepared:
        normalization_mode = source_fit_mode if crop_enabled else "contain"
        vfilters.extend(
            _build_source_normalization_filters(
                src_w,
                src_h,
                crop_position=crop_pos,
                fit_mode=normalization_mode,
                rotation=source_rotation,
                pan_x=source_pan_x,
                pan_y=source_pan_y,
                zoom=source_zoom,
                final_scale=False,
            )
        )
    else:
        vfilters.append("setsar=1")

    # Step 1: Crop to 9:16
    if False and crop_enabled:
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
    vfilters.append(f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=lanczos")
    # Ensure even dimensions (required by most codecs)
    vfilters.append("setsar=1")

    # Step 3: Cinematic grading and subtle color mood motion
    vfilters.extend(_build_look_filters(look_preset, look_strength, look_motion))

    # Step 4: Optional extra tint overlay
    if tint_enabled and tint_color:
        hex_clean = tint_color.lstrip("#")
        try:
            int(hex_clean[0:2], 16)
            int(hex_clean[2:4], 16)
            int(hex_clean[4:6], 16)
        except (ValueError, IndexError):
            logger.warning(f"Invalid tint color '{tint_color}', skipping tint.")
            tint_enabled = False

        if tint_enabled:
            opacity = max(0.0, min(1.0, tint_opacity))
            vfilters.append(
                f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{hex_clean}@{opacity}:t=fill"
            )

    # Step 5: Template-driven permanent text
    text_overlay = _build_text_overlay(
        text_mode=text_mode,
        primary=text_primary,
        secondary=text_secondary,
        highlight_text=text_highlight,
        accent_color=text_accent_color,
        highlight_color=highlight_color,
        text_bold=text_bold,
        text_scale=text_scale,
        top_text_scale=top_text_scale,
        text_box=text_box,
        temp_dir=temp_dir,
    )

    # Step 6: Watermark (channel name)
    if watermark_enabled and channel_name.strip():
        vfilters.append(_build_watermark_filter(channel_name.strip(), channel_position))

    # Step 7: Captions (subtitles)
    if captions_enabled and caption_path and os.path.isfile(caption_path):
        caption_format = (caption_format or Path(caption_path).suffix.lstrip(".")).lower()
        escaped_caption_path = _escape_filter_path(caption_path)
        if caption_format == "ass":
            vfilters.append(f"ass='{escaped_caption_path}'")
        else:
            vfilters.append(f"subtitles='{escaped_caption_path}'")
    elif captions_enabled and (caption_path is None or not os.path.isfile(caption_path or "")):
        logger.warning("Captions enabled but no caption file found. Skipping captions.")

    # ---- Build audio filter chain ----
    # We need to handle: original audio + optional background music
    audio_inputs = []
    audio_filters = []
    input_args = []
    if input_start is not None:
        input_args.extend(["-ss", f"{input_start:.3f}"])
    if input_start is not None or input_end is not None:
        input_args.extend(["-t", f"{duration:.3f}"])
    input_args.extend(["-i", input_path])
    input_count = 1

    if music_enabled and music_path and os.path.isfile(music_path):
        # Add music as second input, loop it
        input_args.extend(["-stream_loop", "-1", "-i", music_path])
        music_idx = input_count
        input_count += 1

        vol = max(0.0, min(1.0, music_volume))
        duck_threshold = 0.08 - (ducking_strength * 0.06)
        duck_ratio = 1.5 + (ducking_strength * 10.5)
        duck_attack = 12 + int((1.0 - ducking_strength) * 40)
        duck_release = 220 + int((1.0 - ducking_strength) * 240)
        duck_makeup = 1.0 + ducking_strength * 0.3

        if has_audio:
            # Mix original audio + music
            if duck_music:
                audio_filters.append(
                    f"[0:a]volume={audio_boost:.2f},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[dry];"
                    f"[{music_idx}:a]volume={vol},atrim=0:{duration},asetpts=PTS-STARTPTS,"
                    "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[bgm];"
                    f"[bgm][dry]sidechaincompress=threshold={duck_threshold:.3f}:ratio={duck_ratio:.2f}:"
                    f"attack={duck_attack}:release={duck_release}:makeup={duck_makeup:.2f}[ducked];"
                    "[dry][ducked]amix=inputs=2:duration=first:dropout_transition=2,"
                    "alimiter=limit=0.95[aout]"
                )
            else:
                audio_filters.append(
                    f"[0:a]volume={audio_boost:.2f},aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[dry];"
                    f"[{music_idx}:a]volume={vol},atrim=0:{duration},asetpts=PTS-STARTPTS,"
                    "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[bgm];"
                    "[dry][bgm]amix=inputs=2:duration=first:dropout_transition=2,"
                    "alimiter=limit=0.95[aout]"
                )
        else:
            # Only music (no original audio)
            audio_filters.append(
                f"[{music_idx}:a]volume={vol},atrim=0:{duration},asetpts=PTS-STARTPTS[aout]"
            )
    elif has_audio:
        # Just pass through original audio
        audio_filters.append(f"[0:a]volume={audio_boost:.2f},alimiter=limit=0.95[aout]")
    # else: no audio at all

    # ---- Combine into final FFmpeg command ----
    vfilter_str = ",".join(vfilters) if vfilters else "null"

    # Build complex filter graph
    filter_parts = []
    filter_parts.append(f"[0:v]{vfilter_str}[vbase]")
    if text_overlay:
        overlay_path = _escape_filter_path(text_overlay["path"])
        filter_parts.append(f"movie='{overlay_path}',format=rgba[text_ov]")
        filter_parts.append(
            f"[vbase][text_ov]overlay="
            f"x={text_overlay['x']}:y={text_overlay['y']}:format=auto[vout]"
        )
    else:
        filter_parts.append("[vbase]null[vout]")
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
    quality_map = {
        "high": {"preset": "slow", "crf": "18"},
        "balanced": {"preset": "medium", "crf": "20"},
        "fast": {"preset": "veryfast", "crf": "23"},
    }
    quality_settings = quality_map.get(export_quality, quality_map["high"])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", quality_settings["preset"],
        "-crf", quality_settings["crf"],
        "-profile:v", "high",
        "-level", "4.1",
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


def build_raw_clip(
    source_videos: list,
    segments: list,
    output_path: str,
    crop_position: float = 0.5,
    source_fit_mode: str = "cover",
    source_rotation: str = "none",
    source_pan_x: float = 0.0,
    source_pan_y: float = 0.0,
    source_zoom: float = 1.0,
    trim_silence: bool = False,
    silence_threshold_db: float = -45.0,
    min_silence_duration: float = 0.35,
    silence_padding: float = 0.1,
    progress_callback=None,
    progress_range: tuple = (1, 20),
):
    """Trim and merge user-selected source segments into one vertical raw clip."""
    ffmpeg = _find_ffmpeg()
    if not source_videos:
        raise ProcessingError("No source videos were provided for raw clip building.")

    normalized_segments = _normalize_segments(source_videos, segments)
    if trim_silence:
        normalized_segments = _expand_segments_by_silence(
            ffmpeg=ffmpeg,
            source_videos=source_videos,
            segments=normalized_segments,
            silence_threshold_db=silence_threshold_db,
            min_silence_duration=min_silence_duration,
            silence_padding=silence_padding,
        )
    if not normalized_segments:
        raise ProcessingError("No valid source segments were provided.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="shorts_raw_builder_")
    segment_files = []

    try:
        start_pct, end_pct = progress_range
        prep_span = max(1.0, (end_pct - start_pct) * 0.8)
        merge_start = start_pct + prep_span
        per_segment_span = prep_span / max(len(normalized_segments), 1)

        for index, segment in enumerate(normalized_segments):
            source = source_videos[segment["video_index"]]
            info = probe_video(source["path"])
            segment_output = os.path.join(temp_dir, f"segment_{index:03d}.mp4")
            segment_filters = ",".join(
                _build_source_normalization_filters(
                    info["width"],
                    info["height"],
                    crop_position=crop_position,
                    fit_mode=source_fit_mode,
                    rotation=source_rotation,
                    pan_x=source_pan_x,
                    pan_y=source_pan_y,
                    zoom=source_zoom,
                )
            )

            cmd = [
                ffmpeg,
                "-y",
                "-ss",
                f"{segment['start']:.3f}",
                "-to",
                f"{segment['end']:.3f}",
                "-i",
                source["path"],
            ]

            if not info["has_audio"]:
                cmd.extend(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=channel_layout=stereo:sample_rate=48000",
                    ]
                )

            cmd.extend(
                [
                    "-vf",
                    segment_filters,
                    "-r",
                    "30",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "superfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                ]
            )

            if info["has_audio"]:
                cmd.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"])
            else:
                cmd.extend(
                    [
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-shortest",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "96k",
                        "-ar",
                        "48000",
                        "-ac",
                        "2",
                    ]
                )

            cmd.extend(["-movflags", "+faststart", segment_output])

            _run_ffmpeg(
                cmd,
                description=f"Preparing source segment {index + 1}/{len(normalized_segments)}",
                duration=segment["end"] - segment["start"],
                progress_callback=progress_callback,
                progress_range=(
                    start_pct + per_segment_span * index,
                    start_pct + per_segment_span * (index + 1),
                ),
            )
            segment_files.append(segment_output)

        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w", encoding="utf-8") as handle:
            for segment_file in segment_files:
                escaped = segment_file.replace("'", "'\\''")
                handle.write(f"file '{escaped}'\n")

        merge_cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            output_path,
        ]
        _run_ffmpeg(
            merge_cmd,
            description="Merging raw clip",
            progress_callback=progress_callback,
            progress_range=(merge_start, end_pct),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _normalize_segments(source_videos: list, segments: list) -> list:
    """Validate and normalize timeline segment input."""
    if not segments:
        normalized = []
        for idx, source in enumerate(source_videos):
            info = probe_video(source["path"])
            normalized.append(
                {
                    "video_index": idx,
                    "start": 0.0,
                    "end": info["duration"],
                }
            )
        return normalized

    normalized = []
    for raw in segments:
        try:
            video_index = int(raw.get("video_index", 0))
            start = float(raw.get("start", 0.0))
            end = float(raw.get("end", 0.0))
        except (TypeError, ValueError):
            raise ProcessingError("One or more source segments are invalid.")

        if video_index < 0 or video_index >= len(source_videos):
            raise ProcessingError("A source segment references a missing video.")

        info = probe_video(source_videos[video_index]["path"])
        start = max(0.0, min(start, info["duration"]))
        end = max(0.0, min(end, info["duration"]))
        if end <= start:
            raise ProcessingError("Each source segment must have an end time after its start time.")

        normalized.append(
            {
                "video_index": video_index,
                "start": start,
                "end": end,
            }
        )

    return normalized


def _expand_segments_by_silence(
    ffmpeg: str,
    source_videos: list,
    segments: list,
    silence_threshold_db: float,
    min_silence_duration: float,
    silence_padding: float,
) -> list:
    """Split segments around detected dead-silent intervals."""
    expanded = []
    silence_cache = {}

    for segment in segments:
        video_index = int(segment["video_index"])
        source = source_videos[video_index]
        info = probe_video(source["path"])

        if not info["has_audio"]:
            expanded.append(segment)
            continue

        if video_index not in silence_cache:
            silence_cache[video_index] = _detect_silence_intervals(
                ffmpeg=ffmpeg,
                input_path=source["path"],
                threshold_db=silence_threshold_db,
                min_duration=min_silence_duration,
                max_duration=info["duration"],
            )

        kept_ranges = _subtract_silence_from_range(
            start=float(segment["start"]),
            end=float(segment["end"]),
            silences=silence_cache[video_index],
            padding=silence_padding,
        )

        if kept_ranges:
            for keep_start, keep_end in kept_ranges:
                expanded.append(
                    {
                        "video_index": video_index,
                        "start": keep_start,
                        "end": keep_end,
                    }
                )
        else:
            expanded.append(segment)

    return expanded


def _detect_silence_intervals(
    ffmpeg: str,
    input_path: str,
    threshold_db: float,
    min_duration: float,
    max_duration: float,
) -> list[tuple[float, float]]:
    """Use ffmpeg silencedetect to find dead-silent intervals."""
    attempts = [
        (threshold_db, min_duration),
        (max(threshold_db + 8.0, -35.0), max(0.2, min_duration * 0.85)),
    ]
    best_intervals = []
    for noise_db, duration in attempts:
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-i",
            input_path,
            "-af",
            f"silencedetect=noise={noise_db:.1f}dB:d={duration:.2f}",
            "-f",
            "null",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stderr = proc.stderr or ""
        silence_start_pattern = re.compile(r"silence_start:\s*([0-9.]+)")
        silence_end_pattern = re.compile(r"silence_end:\s*([0-9.]+)")

        intervals = []
        current_start = None
        for line in stderr.splitlines():
            match_start = silence_start_pattern.search(line)
            if match_start:
                current_start = float(match_start.group(1))
                continue

            match_end = silence_end_pattern.search(line)
            if match_end and current_start is not None:
                end_time = float(match_end.group(1))
                if end_time > current_start:
                    intervals.append((current_start, end_time))
                current_start = None

        if current_start is not None and max_duration > current_start:
            intervals.append((current_start, max_duration))

        if intervals:
            return intervals
        best_intervals = intervals

    return best_intervals


def _subtract_silence_from_range(
    start: float,
    end: float,
    silences: list[tuple[float, float]],
    padding: float,
) -> list[tuple[float, float]]:
    """Keep only the non-silent subranges inside a segment."""
    if end <= start:
        return []

    padded_silences = []
    for silence_start, silence_end in silences:
        trimmed_start = max(start, silence_start + padding)
        trimmed_end = min(end, silence_end - padding)
        if trimmed_end - trimmed_start > 0.05:
            padded_silences.append((trimmed_start, trimmed_end))

    if not padded_silences:
        return [(start, end)]

    keep_ranges = []
    cursor = start
    for silence_start, silence_end in padded_silences:
        if silence_start > cursor:
            keep_ranges.append((cursor, silence_start))
        cursor = max(cursor, silence_end)
    if cursor < end:
        keep_ranges.append((cursor, end))

    return [(seg_start, seg_end) for seg_start, seg_end in keep_ranges if seg_end - seg_start > 0.08]


def _build_source_normalization_filters(
    src_w: int,
    src_h: int,
    crop_position: float,
    fit_mode: str = "cover",
    rotation: str = "none",
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    zoom: float = 1.0,
    final_scale: bool = True,
) -> list:
    """Normalize a source clip into the vertical shorts frame."""
    filters = []
    fit_mode = str(fit_mode or "cover").strip().lower()
    rotation = str(rotation or "none").strip().lower()
    pan_x = _clamp_float(pan_x, -1.0, 1.0)
    pan_y = _clamp_float(pan_y, -1.0, 1.0)
    zoom = _clamp_float(zoom, 0.6, 2.5)

    eff_w, eff_h = src_w, src_h
    if rotation == "cw":
        filters.append("transpose=1")
        eff_w, eff_h = src_h, src_w
    elif rotation == "ccw":
        filters.append("transpose=2")
        eff_w, eff_h = src_h, src_w
    elif rotation == "180":
        filters.append("rotate=PI")

    if fit_mode == "contain":
        base_scale = min(TARGET_WIDTH / eff_w, TARGET_HEIGHT / eff_h)
    else:
        base_scale = max(TARGET_WIDTH / eff_w, TARGET_HEIGHT / eff_h)

    scale_factor = base_scale * zoom
    scaled_w = max(2, int(round(eff_w * scale_factor / 2)) * 2)
    scaled_h = max(2, int(round(eff_h * scale_factor / 2)) * 2)
    filters.append(f"scale={scaled_w}:{scaled_h}:flags=lanczos")

    current_w = scaled_w
    current_h = scaled_h
    if current_w > TARGET_WIDTH:
        x_offset = int(round((current_w - TARGET_WIDTH) * ((pan_x + 1.0) / 2.0)))
        x_offset = max(0, min(current_w - TARGET_WIDTH, x_offset))
        filters.append(f"crop={TARGET_WIDTH}:{current_h}:{x_offset}:0")
        current_w = TARGET_WIDTH

    if current_h > TARGET_HEIGHT:
        if current_h == scaled_h and current_w == TARGET_WIDTH:
            y_offset = int(round((current_h - TARGET_HEIGHT) * ((pan_y + 1.0) / 2.0)))
        else:
            fallback = _clamp_float(crop_position, 0.0, 1.0)
            y_offset = int(round((current_h - TARGET_HEIGHT) * ((pan_y + 1.0) / 2.0 if abs(pan_y) > 0.001 else fallback)))
        y_offset = max(0, min(current_h - TARGET_HEIGHT, y_offset))
        filters.append(f"crop={current_w}:{TARGET_HEIGHT}:0:{y_offset}")
        current_h = TARGET_HEIGHT

    if current_w < TARGET_WIDTH or current_h < TARGET_HEIGHT:
        x_pad = int(round((TARGET_WIDTH - current_w) * ((pan_x + 1.0) / 2.0)))
        y_pad = int(round((TARGET_HEIGHT - current_h) * ((pan_y + 1.0) / 2.0)))
        x_pad = max(0, min(TARGET_WIDTH - current_w, x_pad))
        y_pad = max(0, min(TARGET_HEIGHT - current_h, y_pad))
        filters.append(f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:{x_pad}:{y_pad}:black")
        current_w = TARGET_WIDTH
        current_h = TARGET_HEIGHT

    if final_scale and (current_w != TARGET_WIDTH or current_h != TARGET_HEIGHT):
        filters.append(f"scale={TARGET_WIDTH}:{TARGET_HEIGHT}:flags=lanczos")

    filters.append("setsar=1")
    return filters


def _build_look_filters(preset_name: str, strength: float, motion: float) -> list:
    """Return FFmpeg filters for cinematic grading and subtle mood motion."""
    preset = LOOK_PRESETS.get(preset_name, LOOK_PRESETS["warm_cinematic"])
    eq = preset["eq"]
    pulse = preset["pulse"]

    contrast = eq["contrast"] + (eq["contrast"] - 1.0) * (strength - 1.0)
    if eq["saturation"] == 0.0:
        saturation = 0.0
    else:
        saturation = 1.0 + (eq["saturation"] - 1.0) * strength
    brightness = eq["brightness"] * strength
    gamma = 1.0 + (eq["gamma"] - 1.0) * strength

    contrast_pulse = pulse["contrast"] * motion
    saturation_pulse = pulse["saturation"] * motion
    gamma_pulse = pulse["gamma"] * motion

    filters = [
        "format=yuv420p",
        (
            "eq="
            f"contrast='{contrast:.3f}+{contrast_pulse:.3f}*sin(t*0.55)'"
            f":saturation='{saturation:.3f}+{saturation_pulse:.3f}*sin(t*0.72)'"
            f":brightness='{brightness:.3f}'"
            f":gamma='{gamma:.3f}+{gamma_pulse:.3f}*sin(t*0.31)'"
        ),
    ]

    if preset.get("colorbalance"):
        color_values = []
        for key, value in preset["colorbalance"].items():
            color_values.append(f"{key}={value * strength:.3f}")
        if color_values:
            filters.append("colorbalance=" + ":".join(color_values))

    vignette_strength = 0.08 + preset["vignette"] * strength
    sharpen_strength = max(0.15, preset["sharpen"] * (0.55 + 0.45 * strength))
    filters.append(f"vignette=angle=PI/{max(3.1, 5.2 - vignette_strength * 5.0):.3f}")
    filters.append(f"unsharp=5:5:{sharpen_strength:.3f}:5:5:0.0")

    return filters


def _build_text_filters(
    text_mode: str,
    primary: str,
    secondary: str,
    accent_color: str,
    text_scale: float,
    top_text_scale: float,
    text_box: dict,
) -> list:
    """Return drawtext overlays for the selected cinematic template."""
    accent = _normalize_hex_color(accent_color, "#18D7FF")
    primary_text = primary.strip()
    secondary_text = secondary.strip()
    box = _normalize_text_box(text_box)
    left = int(box["x"] * TARGET_WIDTH)
    top = int(box["y"] * TARGET_HEIGHT)
    width = int(box["w"] * TARGET_WIDTH)
    height = int(box["h"] * TARGET_HEIGHT)
    center_x = left + width // 2

    if text_mode == "none" or not primary_text:
        return []

    filters = []
    if text_mode == "center_title":
        filters.append(
            _drawtext_filter(
                text=_normalize_text_line(primary_text, uppercase=True),
                fontsize=int(74 * text_scale),
                fontcolor=accent,
                x=f"{center_x}-text_w/2",
                y=f"{top + int(height * 0.18)}",
                borderw=4,
                bordercolor="black@0.65",
                shadowx=0,
                shadowy=0,
                shadowcolor="black@0.75",
                line_spacing=12,
                prefer_extended_glyphs=_contains_extended_glyphs(primary_text),
            )
        )
        if secondary_text:
            filters.append(
                _drawtext_filter(
                    text=_normalize_text_line(secondary_text, uppercase=False),
                    fontsize=int(36 * text_scale),
                    fontcolor="white",
                    x=f"{center_x}-text_w/2",
                    y=f"{top + int(height * 0.62)}",
                    borderw=2,
                    bordercolor="black@0.45",
                    shadowx=0,
                    shadowy=0,
                    shadowcolor="black@0.65",
                    line_spacing=10,
                    prefer_extended_glyphs=_contains_extended_glyphs(secondary_text),
                )
            )
    elif text_mode == "premium_subtitle":
        filters.append(
            _drawtext_filter(
                text=_normalize_text_line(primary_text, uppercase=True),
                fontsize=int(48 * text_scale),
                fontcolor=accent,
                x=f"{center_x}-text_w/2",
                y=f"{top + int(height * 0.12)}",
                borderw=3,
                bordercolor="black@0.55",
                shadowx=0,
                shadowy=0,
                shadowcolor="black@0.7",
                line_spacing=8,
                prefer_extended_glyphs=_contains_extended_glyphs(primary_text),
            )
        )
        if secondary_text:
            filters.append(
                _drawtext_filter(
                    text=_normalize_text_line(secondary_text, uppercase=True),
                    fontsize=int(40 * text_scale),
                    fontcolor="white",
                    x=f"{center_x}-text_w/2",
                    y=f"{top + int(height * 0.5)}",
                    borderw=3,
                    bordercolor="black@0.65",
                    shadowx=0,
                    shadowy=0,
                    shadowcolor="black@0.72",
                    line_spacing=8,
                    prefer_extended_glyphs=_contains_extended_glyphs(secondary_text),
                )
            )
    return filters


def _drawtext_filter(
    text: str,
    fontsize: int,
    fontcolor: str,
    x: str,
    y: str,
    borderw: int,
    bordercolor: str,
    shadowx: int,
    shadowy: int,
    shadowcolor: str,
    line_spacing: int,
    prefer_extended_glyphs: bool = False,
) -> str:
    """Build a drawtext filter string."""
    font_arg = _build_drawtext_font_arg(prefer_extended_glyphs)
    return (
        f"drawtext={font_arg}"
        f":text='{_escape_drawtext_text(text)}'"
        f":fontsize={max(16, fontsize)}"
        f":fontcolor={fontcolor}"
        f":x={x}"
        f":y={y}"
        f":borderw={borderw}"
        f":bordercolor={bordercolor}"
        f":shadowx={shadowx}"
        f":shadowy={shadowy}"
        f":shadowcolor={shadowcolor}"
        f":line_spacing={line_spacing}"
        ":text_shaping=1"
    )


def _normalize_text_line(text: str, uppercase: bool) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned.upper() if uppercase else cleaned


def _normalize_text_block(text: str, uppercase: bool) -> str:
    lines = [re.sub(r"\s+", " ", part.strip()) for part in text.replace("\r", "").split("\n")]
    lines = [line for line in lines if line]
    cleaned = r"\n".join(lines[:4])
    return cleaned.upper() if uppercase else cleaned


def _build_text_overlay(
    text_mode: str,
    primary: str,
    secondary: str,
    highlight_text: str,
    accent_color: str,
    highlight_color: str,
    text_bold: bool,
    text_scale: float,
    top_text_scale: float,
    text_box: dict,
    temp_dir: str,
):
    """Render permanent template text into a transparent PNG for reliable emoji support."""
    if text_mode == "none":
        return None

    primary_text = _normalize_text_block(primary, uppercase=False)
    if not primary_text:
        return None

    os.makedirs(temp_dir, exist_ok=True)
    box = _normalize_text_box(text_box)
    width = max(240, int(box["w"] * TARGET_WIDTH))
    height = max(96, int(box["h"] * TARGET_HEIGHT))
    left = int(box["x"] * TARGET_WIDTH)
    top = int(box["y"] * TARGET_HEIGHT)
    if text_mode == "center_title":
        primary_size = max(28, int(74 * text_scale))
        secondary_size = max(18, int(36 * text_scale))
    elif text_mode == "premium_subtitle":
        primary_size = max(24, int(48 * text_scale))
        secondary_size = max(20, int(40 * text_scale))
    else:
        primary_size = max(22, int(34 * _clamp_float(text_scale * top_text_scale, 0.7, 2.4)))
        secondary_size = 0
    accent = _normalize_hex_color(highlight_color, "#FF7B47")
    accent_text = _normalize_hex_color(accent_color, "#18D7FF")
    highlight_tokens = _build_highlight_token_set(highlight_text)
    image_path = os.path.join(temp_dir, "text_overlay.png")
    secondary_text = secondary.strip()

    try:
        _render_text_overlay_png(
            text_mode=text_mode,
            primary_text=primary_text,
            secondary_text=secondary_text,
            output_path=image_path,
            width=width,
            height=height,
            primary_font_size=primary_size,
            secondary_font_size=secondary_size,
            accent_color=accent_text,
            highlight_color=accent,
            highlight_tokens=highlight_tokens,
            bold=text_bold,
        )
    except Exception as exc:
        logger.warning("Template text overlay render failed, falling back to no permanent text: %s", exc)
        return None

    return {"path": image_path, "x": left, "y": top}


def _render_text_overlay_png(
    text_mode: str,
    primary_text: str,
    secondary_text: str,
    output_path: str,
    width: int,
    height: int,
    primary_font_size: int,
    secondary_font_size: int,
    accent_color: str,
    highlight_color: str,
    highlight_tokens: set[str],
    bold: bool,
):
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    primary_font = _load_overlay_font(_find_bold_font_file() if bold else _find_regular_font_file(), primary_font_size)
    secondary_font = _load_overlay_font(_find_bold_font_file() if bold else _find_regular_font_file(), secondary_font_size or max(18, primary_font_size // 2))
    emoji_font = _load_emoji_overlay_font(_find_overlay_emoji_font_file())

    if text_mode == "top_commentary":
        _render_wrapped_line_block(
            image, draw, primary_text, 0, 0, width, height, primary_font, emoji_font,
            highlight_tokens=highlight_tokens, highlight_color=highlight_color,
            normal_color="white", line_gap=max(8, int(primary_font_size * 0.18))
        )
    else:
        top_padding = max(8, int(height * 0.12))
        primary_height = max(24, int(height * (0.34 if text_mode == "center_title" else 0.28)))
        secondary_height = max(20, int(height * 0.2))
        _render_wrapped_line_block(
            image, draw, _normalize_text_line(primary_text, uppercase=text_mode != "top_commentary"),
            0, top_padding, width, primary_height, primary_font, emoji_font,
            highlight_tokens=highlight_tokens, highlight_color=highlight_color,
            normal_color=accent_color, line_gap=max(6, int(primary_font_size * 0.14))
        )
        if secondary_text:
            _render_wrapped_line_block(
                image, draw, _normalize_text_line(secondary_text, uppercase=text_mode == "premium_subtitle"),
                0, top_padding + int(height * (0.44 if text_mode == "center_title" else 0.46)),
                width, secondary_height, secondary_font, emoji_font,
                highlight_tokens=set(), highlight_color=highlight_color,
                normal_color="white", line_gap=max(6, int(secondary_font_size * 0.14))
            )

    image.save(output_path)


def _render_wrapped_line_block(
    image,
    draw,
    text: str,
    left: int,
    top: int,
    width: int,
    height: int,
    regular_font,
    emoji_font,
    highlight_tokens: set[str],
    highlight_color: str,
    normal_color: str,
    line_gap: int,
):
    max_text_width = max(120, width - 28)
    lines = _wrap_mixed_text(text, draw, regular_font, emoji_font, max_text_width)
    if not lines:
        lines = [text]

    measured_lines = [_measure_mixed_text(line, draw, regular_font, emoji_font, highlight_tokens) for line in lines]
    total_height = sum(item["height"] for item in measured_lines) + line_gap * max(0, len(measured_lines) - 1)
    start_y = top + max(0, (height - total_height) // 2)

    current_y = start_y
    for line, metrics in zip(lines, measured_lines):
        current_x = left + max(0, (width - metrics["width"]) // 2)
        for run in _split_mixed_runs(line, highlight_tokens):
            if run["emoji"] and emoji_font:
                emoji_image = _render_emoji_run_image(run["text"], emoji_font, regular_font.size)
                run_width, run_height = emoji_image.size
                run_y = current_y + max(0, (metrics["height"] - run_height) // 2)
                image.alpha_composite(emoji_image, (current_x, run_y))
            else:
                run_font = regular_font
                bbox = draw.textbbox((0, 0), run["text"], font=run_font)
                run_width = max(0, bbox[2] - bbox[0])
                run_height = max(0, bbox[3] - bbox[1])
                run_y = current_y + max(0, (metrics["height"] - run_height) // 2)
                draw.text((current_x, run_y + 2), run["text"], font=run_font, fill=(0, 0, 0, 180))
                draw.text(
                    (current_x, run_y),
                    run["text"],
                    font=run_font,
                    fill=highlight_color if run.get("highlight") else normal_color,
                    stroke_width=max(1, regular_font.size // 16),
                    stroke_fill=(0, 0, 0, 170),
                )
            current_x += run_width
        current_y += metrics["height"] + line_gap


def _wrap_mixed_text(text: str, draw, regular_font, emoji_font, max_width: int) -> list[str]:
    lines = []
    for raw_line in text.split(r"\n"):
        words = raw_line.split()
        if not words:
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if _measure_mixed_text(candidate, draw, regular_font, emoji_font)["width"] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines[:4]


def _measure_mixed_text(text: str, draw, regular_font, emoji_font, highlight_tokens: set[str] | None = None) -> dict:
    width = 0
    height = 0
    for run in _split_mixed_runs(text, highlight_tokens or set()):
        if run["emoji"] and emoji_font:
            emoji_image = _render_emoji_run_image(run["text"], emoji_font, regular_font.size)
            width += emoji_image.size[0]
            height = max(height, emoji_image.size[1])
        else:
            bbox = draw.textbbox((0, 0), run["text"], font=regular_font)
            width += max(0, bbox[2] - bbox[0])
            height = max(height, max(0, bbox[3] - bbox[1]))
    return {"width": width, "height": height or regular_font.size}


def _split_mixed_runs(text: str, highlight_tokens: set[str]) -> list[dict]:
    runs = []
    parts = re.split(r"(\s+)", text)
    for part in parts:
        if not part:
            continue
        if part.isspace():
            runs.append({"text": part, "emoji": False, "highlight": False})
            continue
        normalized = re.sub(r"[^\w]+", "", part, flags=re.UNICODE).lower()
        token_highlight = bool(normalized and normalized in highlight_tokens)
        current = []
        current_is_emoji = None
        for char in part:
            char_is_emoji = _is_emoji_like_char(char)
            if current_is_emoji is None or char_is_emoji == current_is_emoji:
                current.append(char)
                current_is_emoji = char_is_emoji
            else:
                runs.append({"text": "".join(current), "emoji": current_is_emoji, "highlight": token_highlight and not current_is_emoji})
                current = [char]
                current_is_emoji = char_is_emoji
        if current:
            runs.append({"text": "".join(current), "emoji": current_is_emoji, "highlight": token_highlight and not current_is_emoji})
    return runs


def _is_emoji_like_char(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    if char in {"\u200d", "\ufe0f"}:
        return True
    return (
        0x1F300 <= codepoint <= 0x1FAFF
        or 0x2600 <= codepoint <= 0x27BF
        or unicodedata.category(char) == "So"
    )


def _load_overlay_font(font_path: str | None, font_size: int):
    if font_path and os.path.isfile(font_path):
        return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


def _load_emoji_overlay_font(font_path: str | None):
    if font_path and os.path.isfile(font_path):
        for supported_size in (109,):
            try:
                return ImageFont.truetype(font_path, supported_size)
            except OSError:
                continue
    return None


def _render_emoji_run_image(text: str, emoji_font, target_height: int) -> Image.Image:
    dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    probe = ImageDraw.Draw(dummy)
    bbox = probe.textbbox((0, 0), text, font=emoji_font, embedded_color=True)
    width = max(1, bbox[2] - bbox[0] + 8)
    height = max(1, bbox[3] - bbox[1] + 8)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((4 - bbox[0], 4 - bbox[1]), text, font=emoji_font, embedded_color=True)
    cropped = image.getbbox()
    if cropped:
        image = image.crop(cropped)
    if image.height <= 0:
        return image
    scale = max(0.1, target_height / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.LANCZOS,
    )
    return resized


def _find_regular_font_file() -> str | None:
    for font_path in [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if os.path.isfile(font_path):
            return font_path
    return None


def _find_bold_font_file() -> str | None:
    for font_path in [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\seguisb.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]:
        if os.path.isfile(font_path):
            return font_path
    return _find_regular_font_file()


def _find_overlay_emoji_font_file() -> str | None:
    candidates = [
        str(BUNDLED_EMOJI_FONT),
        r"C:\Windows\Fonts\seguiemj.ttf",
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
    ]
    for font_path in candidates:
        if os.path.isfile(font_path):
            return font_path
    return None


def _build_highlight_token_set(text: str) -> set[str]:
    tokens = set()
    for token in re.split(r"[\s,]+", str(text or "")):
        normalized = re.sub(r"[^\w]+", "", token, flags=re.UNICODE).lower()
        if normalized:
            tokens.add(normalized)
    return tokens


def _build_watermark_filter(channel_name: str, position: str) -> str:
    safe_name = _escape_drawtext_text(channel_name.strip().replace("@", "", 1 if channel_name.startswith("@") else 0))
    safe_name = "@" + safe_name.lstrip("@")
    position = str(position or "lower_left_overlay").strip().lower()
    font_path = _find_bold_font_file()
    font_clause = f":fontfile='{_escape_filter_path(font_path)}'" if font_path else ""

    if position == "bottom_center":
        x_expr = "(w-text_w)/2"
        y_expr = "h-th-60"
        fontsize = 28
        opacity = "white@0.6"
    elif position == "center_overlay":
        x_expr = "(w-text_w)/2"
        y_expr = "h*0.54"
        fontsize = 34
        opacity = "white@0.92"
    else:
        x_expr = "w*0.12"
        y_expr = "h*0.73"
        fontsize = 32
        opacity = "white@0.95"

    return (
        f"drawtext=text='{safe_name}'"
        f"{font_clause}"
        f":fontsize={fontsize}"
        f":fontcolor={opacity}"
        f":x={x_expr}"
        f":y={y_expr}"
        f":borderw=2"
        f":bordercolor=black@0.55"
        f":shadowx=0"
        f":shadowy=2"
        f":shadowcolor=black@0.72"
    )


def _contains_extended_glyphs(text: str) -> bool:
    for char in str(text or ""):
        if ord(char) > 0x7F:
            return True
        if unicodedata.category(char) in {"So", "Sk"}:
            return True
    return False


def _build_drawtext_font_arg(prefer_extended_glyphs: bool) -> str:
    """Return a font clause that gives drawtext a better Unicode fallback path."""
    font_file = _find_font_file(prefer_extended_glyphs)
    if font_file:
        return f"fontfile='{_escape_filter_path(font_file)}'"

    family = EXTENDED_DRAW_FONT_FAMILIES if prefer_extended_glyphs else DEFAULT_DRAW_FONT_FAMILIES
    return f"font='{_escape_drawtext_value(family)}'"


def _find_font_file(prefer_extended_glyphs: bool) -> str | None:
    if prefer_extended_glyphs:
        return None

    common_fonts = [
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for font_path in common_fonts:
        if os.path.isfile(font_path):
            return font_path
    return None


def _escape_drawtext_text(text: str) -> str:
    escaped = text.replace("\\", r"\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(",", r"\,")
    escaped = escaped.replace("%", r"\%")
    escaped = escaped.replace("[", r"\[")
    escaped = escaped.replace("]", r"\]")
    return escaped


def _escape_drawtext_value(text: str) -> str:
    escaped = str(text or "").replace("\\", r"\\")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace(",", r"\,")
    return escaped


def _escape_filter_path(path: str) -> str:
    """Escape a filesystem path for use in FFmpeg filter arguments."""
    escaped = path.replace("\\", "/")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace("[", r"\[")
    escaped = escaped.replace("]", r"\]")
    escaped = escaped.replace(",", r"\,")
    return escaped


def _normalize_hex_color(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
        return text
    return fallback


def _normalize_text_box(value) -> dict:
    fallback = {"x": 0.14, "y": 0.38, "w": 0.72, "h": 0.2}
    if not isinstance(value, dict):
        return fallback

    x = _clamp_float(value.get("x", fallback["x"]), 0.0, 0.88)
    y = _clamp_float(value.get("y", fallback["y"]), 0.0, 0.94)
    w = _clamp_float(value.get("w", fallback["w"]), 0.12, 1.0 - x)
    h = _clamp_float(value.get("h", fallback["h"]), 0.08, 1.0 - y)
    return {"x": x, "y": y, "w": w, "h": h}


def _clamp_float(value, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))
