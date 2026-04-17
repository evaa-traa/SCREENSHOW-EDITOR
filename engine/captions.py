"""
Caption generation helpers.

This module handles:
- Groq Whisper transcription with optional auto language detection
- SRT generation for compatibility
- ASS subtitle generation with animated word highlighting
- Converting custom SRT files into styled ASS overlays
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from copy import deepcopy
from typing import Dict, List, Optional

import requests

from engine.processor import ProcessingError, extract_audio

logger = logging.getLogger("ShortsEditor.Captions")

GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
PLAY_RES_X = 1080
PLAY_RES_Y = 1920

DEFAULT_CAPION_BOX = {
    "x": 0.08,
    "y": 0.72,
    "w": 0.84,
    "h": 0.18,
}

STYLE_PRESETS = {
    "reels": {
        "font_name": "Arial",
        "font_size": 72,
        "primary_color": "#FFFFFF",
        "active_color": "#18D7FF",
        "outline_color": "#000000",
        "back_color": "#000000",
        "outline": 6,
        "shadow": 0,
        "bold": True,
        "uppercase": True,
        "spacing": 1.0,
        "line_spacing": 18,
        "active_scale": 112,
        "pop_ms": 130,
        "max_words": 4,
    },
    "clean": {
        "font_name": "Arial",
        "font_size": 60,
        "primary_color": "#FFFFFF",
        "active_color": "#FFD54A",
        "outline_color": "#111111",
        "back_color": "#000000",
        "outline": 4,
        "shadow": 1,
        "bold": True,
        "uppercase": False,
        "spacing": 0.3,
        "line_spacing": 14,
        "active_scale": 106,
        "pop_ms": 110,
        "max_words": 5,
    },
    "impact": {
        "font_name": "Arial",
        "font_size": 82,
        "primary_color": "#FFFFFF",
        "active_color": "#FF9F1C",
        "outline_color": "#000000",
        "back_color": "#000000",
        "outline": 7,
        "shadow": 0,
        "bold": True,
        "uppercase": True,
        "spacing": 1.2,
        "line_spacing": 20,
        "active_scale": 116,
        "pop_ms": 150,
        "max_words": 3,
    },
}


def normalize_caption_settings(settings: Optional[dict] = None) -> dict:
    """Return sanitized caption layout/style settings."""
    incoming = settings or {}
    preset_name = str(incoming.get("preset", "reels")).strip().lower()
    preset = deepcopy(STYLE_PRESETS.get(preset_name, STYLE_PRESETS["reels"]))

    font_size = int(_clamp(incoming.get("font_size", preset["font_size"]), 28, 160))
    max_words = int(_clamp(incoming.get("max_words", preset["max_words"]), 1, 8))

    box = incoming.get("box") or {}
    x = _clamp(box.get("x", DEFAULT_CAPION_BOX["x"]), 0.0, 0.9)
    y = _clamp(box.get("y", DEFAULT_CAPION_BOX["y"]), 0.0, 0.94)
    w = _clamp(box.get("w", DEFAULT_CAPION_BOX["w"]), 0.12, 1.0 - x)
    h = _clamp(box.get("h", DEFAULT_CAPION_BOX["h"]), 0.08, 1.0 - y)

    preset.update(
        {
            "preset": preset_name,
            "font_size": font_size,
            "max_words": max_words,
            "box": {
                "x": round(x, 4),
                "y": round(y, 4),
                "w": round(w, 4),
                "h": round(h, 4),
            },
        }
    )
    return preset


def transcribe_with_groq(
    video_path: str,
    api_key: str,
    language: str = "auto",
) -> dict:
    """Transcribe a video file with Groq Whisper and return verbose JSON."""
    if not api_key or api_key.strip() == "" or api_key == "your_groq_api_key_here":
        raise ProcessingError(
            "Groq API key not set.\n"
            "Please add your key to the .env file:\n"
            "GROQ_API_KEY=gsk_your_key_here"
        )

    logger.info("Extracting audio for transcription...")
    wav_path = None
    try:
        wav_dir = tempfile.mkdtemp(prefix="shorts_captions_")
        wav_path = os.path.join(wav_dir, "audio.wav")
        extract_audio(video_path, wav_path)

        if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            raise ProcessingError("Audio extraction produced an empty file.")

        file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        if file_size_mb > 25:
            logger.warning(
                "Audio file is %.1fMB, compressing before upload.",
                file_size_mb,
            )
            mp3_path = os.path.join(wav_dir, "audio.mp3")
            _compress_audio(wav_path, mp3_path)
            upload_path = mp3_path
        else:
            upload_path = wav_path

        logger.info("Sending audio to Groq Whisper (language=%s)...", language)
        with open(upload_path, "rb") as audio_file:
            data = [
                ("model", "whisper-large-v3-turbo"),
                ("response_format", "verbose_json"),
                ("temperature", "0"),
                ("timestamp_granularities[]", "segment"),
                ("timestamp_granularities[]", "word"),
            ]
            if language and language != "auto":
                data.append(("language", language))

            response = requests.post(
                GROQ_TRANSCRIPTION_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(upload_path), audio_file)},
                data=data,
                timeout=180,
            )

        if response.status_code != 200:
            error_msg = response.text[:500]
            raise ProcessingError(
                f"Groq Whisper API error (HTTP {response.status_code}):\n{error_msg}"
            )

        result = response.json()
        if not result.get("segments") and not result.get("text"):
            raise ProcessingError(
                "Groq Whisper returned no transcription.\n"
                "The audio may be too short, silent, or unsupported."
            )

        return result
    finally:
        if wav_path:
            try:
                import shutil

                shutil.rmtree(os.path.dirname(wav_path), ignore_errors=True)
            except Exception:
                pass


def generate_srt_groq(
    video_path: str,
    output_srt_path: str,
    api_key: str,
    language: str = "auto",
) -> str:
    """Generate a classic SRT subtitle file using Groq Whisper."""
    result = transcribe_with_groq(video_path, api_key, language=language)
    segments = result.get("segments", [])

    if segments:
        srt_content = _segments_to_srt(segments)
    else:
        text = result.get("text", "").strip()
        duration = float(result.get("duration", 10.0))
        srt_content = (
            "1\n"
            f"00:00:00,000 --> {_format_srt_timestamp(duration)}\n"
            f"{text}\n\n"
        )

    with open(output_srt_path, "w", encoding="utf-8") as handle:
        handle.write(srt_content)

    return output_srt_path


def generate_ass_groq(
    video_path: str,
    output_ass_path: str,
    api_key: str,
    language: str = "auto",
    settings: Optional[dict] = None,
) -> str:
    """Generate animated ASS subtitles from Groq Whisper output."""
    result = transcribe_with_groq(video_path, api_key, language=language)
    caption_settings = normalize_caption_settings(settings)
    ass_content = build_ass_from_transcription(result, caption_settings)

    with open(output_ass_path, "w", encoding="utf-8") as handle:
        handle.write(ass_content)

    logger.info("ASS captions generated: %s", output_ass_path)
    return output_ass_path


def convert_srt_to_ass(
    srt_path: str,
    output_ass_path: str,
    settings: Optional[dict] = None,
) -> str:
    """Convert an SRT file into a styled ASS subtitle file."""
    cues = _parse_srt_file(srt_path)
    if not cues:
        raise ProcessingError("Uploaded SRT file is empty or invalid.")

    ass_content = build_ass_from_srt(cues, normalize_caption_settings(settings))
    with open(output_ass_path, "w", encoding="utf-8") as handle:
        handle.write(ass_content)

    logger.info("Converted SRT to ASS: %s", output_ass_path)
    return output_ass_path


def build_ass_from_transcription(result: dict, settings: Optional[dict] = None) -> str:
    """Build an ASS subtitle document from Groq verbose JSON."""
    caption_settings = normalize_caption_settings(settings)
    words = _normalize_word_items(result)
    if words:
        events = _build_word_highlight_events(words, caption_settings)
    else:
        segments = result.get("segments", [])
        events = _build_segment_events(segments, caption_settings)

    if not events:
        text = result.get("text", "").strip()
        if text:
            fallback_segment = [{"start": 0.0, "end": float(result.get("duration", 5.0)), "text": text}]
            events = _build_segment_events(fallback_segment, caption_settings)

    if not events:
        raise ProcessingError("Could not build subtitle events from the transcription.")

    return _build_ass_document(events, caption_settings)


def build_ass_from_srt(cues: List[dict], settings: Optional[dict] = None) -> str:
    """Build a styled ASS document from SRT cues."""
    caption_settings = normalize_caption_settings(settings)
    events = []
    for cue in cues:
        wrapped = _wrap_plain_text(cue["text"], caption_settings)
        text = (
            f"{_box_override(caption_settings)}"
            f"{_cue_intro_override(caption_settings)}"
            f"{wrapped}"
        )
        events.append(
            {
                "start": cue["start"],
                "end": cue["end"],
                "style": "Caption",
                "text": text,
            }
        )

    return _build_ass_document(events, caption_settings)


def validate_srt_file(srt_path: str) -> bool:
    """Basic validation of an SRT file."""
    if not os.path.isfile(srt_path):
        return False

    try:
        with open(srt_path, "r", encoding="utf-8") as handle:
            content = handle.read(2000)
    except Exception:
        return False

    if not content.strip():
        return False

    return bool(
        re.search(
            r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}",
            content,
        )
    )


def _compress_audio(input_wav: str, output_mp3: str):
    """Compress WAV to MP3 to fit within Groq's 25MB limit."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ProcessingError("FFmpeg not found, needed to compress audio.")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        input_wav,
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_mp3,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    if result.returncode != 0:
        raise ProcessingError(f"Audio compression failed:\n{result.stderr[:500]}")


def _normalize_word_items(result: dict) -> List[dict]:
    """Return normalized word timing items from a Groq transcription result."""
    raw_words = result.get("words") or []
    words = []

    for raw in raw_words:
        text = str(raw.get("word", "")).strip()
        if not text:
            continue
        start = _safe_float(raw.get("start"))
        end = _safe_float(raw.get("end"))
        if start is None or end is None or end <= start:
            continue
        words.append({"text": text, "start": start, "end": end})

    if words:
        return words

    for segment in result.get("segments", []):
        seg_words = segment.get("words") or []
        for raw in seg_words:
            text = str(raw.get("word", "")).strip()
            start = _safe_float(raw.get("start"))
            end = _safe_float(raw.get("end"))
            if not text or start is None or end is None or end <= start:
                continue
            words.append({"text": text, "start": start, "end": end})

    return words


def _build_word_highlight_events(words: List[dict], settings: dict) -> List[dict]:
    """Build one ASS event per spoken word, highlighting the active word."""
    chunks = _chunk_words(words, settings)
    events = []
    active_color = _hex_to_ass_color(settings["active_color"])
    active_scale = int(settings["active_scale"])
    pop_ms = int(settings["pop_ms"])

    for chunk in chunks:
        wrapped_lines = _wrap_word_items(chunk, settings)
        for idx, word in enumerate(chunk):
            if idx + 1 < len(chunk):
                event_end = max(chunk[idx + 1]["start"], word["end"])
            else:
                event_end = word["end"] + 0.06

            line_text = []
            for line in wrapped_lines:
                parts = []
                for item in line:
                    token = item["display"]
                    if item["chunk_index"] == idx:
                        token = (
                            f"{{\\c{active_color}\\fscx{active_scale}\\fscy{active_scale}"
                            f"\\t(0,{pop_ms},\\fscx100\\fscy100)}}"
                            f"{token}{{\\r}}"
                        )
                    parts.append(token)
                line_text.append(" ".join(parts))

            joined_lines = r"\N".join(line_text)
            text = f"{_box_override(settings)}{joined_lines}"
            events.append(
                {
                    "start": word["start"],
                    "end": max(event_end, word["start"] + 0.04),
                    "style": "Caption",
                    "text": text,
                }
            )

    return events


def _build_segment_events(segments: List[dict], settings: dict) -> List[dict]:
    """Fallback event builder when word-level timestamps are unavailable."""
    events = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = _safe_float(segment.get("start"))
        end = _safe_float(segment.get("end"))
        if start is None or end is None or end <= start:
            continue

        wrapped = _wrap_plain_text(text, settings)
        events.append(
            {
                "start": start,
                "end": end,
                "style": "Caption",
                "text": f"{_box_override(settings)}{_cue_intro_override(settings)}{wrapped}",
            }
        )

    return events


def _build_ass_document(events: List[dict], settings: dict) -> str:
    """Assemble a full ASS document."""
    left = int(settings["box"]["x"] * PLAY_RES_X)
    top = int(settings["box"]["y"] * PLAY_RES_Y)
    right_margin = int((1 - settings["box"]["x"] - settings["box"]["w"]) * PLAY_RES_X)
    bottom_margin = int((1 - settings["box"]["y"] - settings["box"]["h"]) * PLAY_RES_Y)

    primary = _hex_to_ass_color(settings["primary_color"])
    secondary = _hex_to_ass_color(settings["active_color"])
    outline = _hex_to_ass_color(settings["outline_color"])
    back = _hex_to_ass_color(settings["back_color"], alpha=0x64)
    bold = -1 if settings.get("bold") else 0

    header = [
        "[Script Info]",
        "Title: ShortsEditor Animated Captions",
        "ScriptType: v4.00+",
        f"PlayResX: {PLAY_RES_X}",
        f"PlayResY: {PLAY_RES_Y}",
        "ScaledBorderAndShadow: yes",
        "WrapStyle: 2",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            "Style: Caption,"
            f"{settings['font_name']},{settings['font_size']},{primary},{secondary},"
            f"{outline},{back},{bold},0,0,0,100,100,{settings['spacing']},0,1,"
            f"{settings['outline']},{settings['shadow']},8,{left},{right_margin},"
            f"{max(top, bottom_margin)},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    lines = header[:]
    for event in events:
        lines.append(
            "Dialogue: 0,"
            f"{_format_ass_timestamp(event['start'])},"
            f"{_format_ass_timestamp(event['end'])},"
            f"{event['style']},,0,0,0,,{event['text']}"
        )

    return "\n".join(lines) + "\n"


def _chunk_words(words: List[dict], settings: dict) -> List[List[dict]]:
    """Group words into short caption bursts that fit common reels pacing."""
    max_words = int(settings["max_words"])
    pause_threshold = 0.42
    max_duration = 2.2
    chunks = []
    current = []

    for word in words:
        if not current:
            current = [word]
            continue

        previous = current[-1]
        gap = max(0.0, word["start"] - previous["end"])
        duration = word["end"] - current[0]["start"]
        previous_text = previous["text"]

        if (
            len(current) >= max_words
            or gap >= pause_threshold
            or duration >= max_duration
            or previous_text.endswith((".", "?", "!", ";", ":"))
        ):
            chunks.append(current)
            current = [word]
        else:
            current.append(word)

    if current:
        chunks.append(current)

    return chunks


def _wrap_word_items(chunk: List[dict], settings: dict) -> List[List[dict]]:
    """Wrap caption words into one or two lines based on box width."""
    display_items = []
    for idx, word in enumerate(chunk):
        display_items.append(
            {
                "chunk_index": idx,
                "display": _display_text(word["text"], settings),
            }
        )

    max_chars = _max_chars_per_line(settings)
    lines: List[List[dict]] = [[]]
    for item in display_items:
        current_line = lines[-1]
        candidate = current_line + [item]
        if current_line and len(_plain_line(candidate)) > max_chars and len(lines) < 2:
            lines.append([item])
        else:
            current_line.append(item)

    if any(not line for line in lines):
        lines = [line for line in lines if line]

    return lines


def _wrap_plain_text(text: str, settings: dict) -> str:
    """Wrap plain cue text into one or two lines for the caption box."""
    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    if not tokens:
        return ""

    converted = [{"display": _display_text(token, settings)} for token in tokens]
    max_chars = _max_chars_per_line(settings)
    lines: List[List[dict]] = [[]]
    for token in converted:
        current_line = lines[-1]
        candidate = current_line + [token]
        if current_line and len(_plain_line(candidate)) > max_chars and len(lines) < 2:
            lines.append([token])
        else:
            current_line.append(token)

    return "\\N".join(_plain_line(line) for line in lines if line)


def _box_override(settings: dict) -> str:
    """ASS override that pins captions to the chosen preview rectangle."""
    left = int(settings["box"]["x"] * PLAY_RES_X)
    top = int(settings["box"]["y"] * PLAY_RES_Y)
    width = int(settings["box"]["w"] * PLAY_RES_X)
    height = int(settings["box"]["h"] * PLAY_RES_Y)
    center_x = left + width // 2
    top_y = top + int(settings["font_size"] * 0.15)
    right = left + width
    bottom = top + height
    return f"{{\\an8\\pos({center_x},{top_y})\\clip({left},{top},{right},{bottom})}}"


def _cue_intro_override(settings: dict) -> str:
    """A small pop-in transform for non-word-timed caption cues."""
    return "{\\fad(50,80)\\fscx96\\fscy96\\t(0,120,\\fscx100\\fscy100)}"


def _parse_srt_file(srt_path: str) -> List[dict]:
    """Parse SRT blocks into cue dictionaries."""
    with open(srt_path, "r", encoding="utf-8-sig") as handle:
        content = handle.read()

    blocks = re.split(r"\r?\n\r?\n+", content.strip())
    cues = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        if "-->" in lines[0]:
            timing_line = lines[0]
            text_lines = lines[1:]
        elif len(lines) >= 2 and "-->" in lines[1]:
            timing_line = lines[1]
            text_lines = lines[2:]
        else:
            continue

        start_text, end_text = [part.strip() for part in timing_line.split("-->", 1)]
        start = _parse_srt_timestamp(start_text)
        end = _parse_srt_timestamp(end_text)
        if start is None or end is None or end <= start:
            continue

        text = " ".join(part.strip() for part in text_lines if part.strip())
        if text:
            cues.append({"start": start, "end": end, "text": text})

    return cues


def _segments_to_srt(segments: list) -> str:
    """Convert Whisper segments to SRT content."""
    srt_lines = []
    index = 1
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        start = _safe_float(segment.get("start"))
        end = _safe_float(segment.get("end"))
        if not text or start is None or end is None or end <= start:
            continue

        srt_lines.append(
            f"{index}\n"
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
            f"{text}\n"
        )
        index += 1

    return "\n".join(srt_lines)


def _format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS,mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        secs += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_ass_timestamp(seconds: float) -> str:
    """Convert seconds to H:MM:SS.cc used by ASS subtitles."""
    total_centis = int(round(seconds * 100))
    hours = total_centis // 360000
    minutes = (total_centis % 360000) // 6000
    secs = (total_centis % 6000) // 100
    centis = total_centis % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _parse_srt_timestamp(value: str) -> Optional[float]:
    """Parse an SRT timestamp into seconds."""
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", value)
    if not match:
        return None
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


def _display_text(text: str, settings: dict) -> str:
    """Apply preset casing and escape ASS special characters."""
    display = text.upper() if settings.get("uppercase") else text
    return _escape_ass_text(display)


def _escape_ass_text(text: str) -> str:
    """Escape special characters in ASS dialogue text."""
    escaped = text.replace("\\", r"\\")
    escaped = escaped.replace("{", r"\{").replace("}", r"\}")
    return escaped.replace("\n", r"\N")


def _plain_line(items: List[dict]) -> str:
    return " ".join(item["display"] for item in items)


def _max_chars_per_line(settings: dict) -> int:
    box_width_px = max(int(settings["box"]["w"] * PLAY_RES_X), 120)
    font_size = max(int(settings["font_size"]), 1)
    approx_char_width = max(font_size * 0.58, 14)
    return max(8, int(box_width_px / approx_char_width))


def _hex_to_ass_color(hex_color: str, alpha: int = 0x00) -> str:
    """Convert #RRGGBB into ASS color format &HAABBGGRR."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        value = "FFFFFF"
    rr = value[0:2]
    gg = value[2:4]
    bb = value[4:6]
    return f"&H{alpha:02X}{bb}{gg}{rr}"


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value, low, high):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = low
    return max(low, min(high, numeric))
