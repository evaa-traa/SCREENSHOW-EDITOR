"""
Caption generation module — Groq Whisper API for Hindi/multilingual transcription.

Supports:
- Auto-generate subtitles via Groq's whisper-large-v3-turbo
- Custom SRT file upload (user-provided)
- Graceful fallback if API is unavailable
"""

import logging
import os
import tempfile

import requests

from engine.processor import extract_audio, ProcessingError

logger = logging.getLogger("ShortsEditor.Captions")

# Groq Whisper API endpoint
GROQ_TRANSCRIPTION_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


def generate_srt_groq(
    video_path: str,
    output_srt_path: str,
    api_key: str,
    language: str = "hi",
) -> str:
    """
    Generate an SRT subtitle file using Groq's Whisper API.

    Parameters
    ----------
    video_path : str
        Path to the source video / audio file.
    output_srt_path : str
        Where to save the generated .srt file.
    api_key : str
        Groq API key.
    language : str
        Language code (e.g., 'hi' for Hindi, 'en' for English).

    Returns
    -------
    str
        Path to the generated SRT file.

    Raises
    ------
    ProcessingError
        If transcription fails.
    """
    if not api_key or api_key.strip() == "" or api_key == "your_groq_api_key_here":
        raise ProcessingError(
            "Groq API key not set.\n"
            "Please add your key to the .env file:\n"
            "GROQ_API_KEY=gsk_your_key_here"
        )

    # Step 1: Extract audio as WAV
    logger.info("Extracting audio for transcription...")
    wav_path = None
    try:
        wav_dir = tempfile.mkdtemp(prefix="shorts_captions_")
        wav_path = os.path.join(wav_dir, "audio.wav")
        extract_audio(video_path, wav_path)

        if not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
            raise ProcessingError("Audio extraction produced an empty file.")

        # Check file size — Groq has a 25MB limit
        file_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
        if file_size_mb > 25:
            logger.warning(f"Audio file is {file_size_mb:.1f}MB (limit 25MB). "
                           "Compressing to mp3...")
            mp3_path = os.path.join(wav_dir, "audio.mp3")
            _compress_audio(wav_path, mp3_path)
            upload_path = mp3_path
        else:
            upload_path = wav_path

        # Step 2: Send to Groq Whisper API
        logger.info(f"Sending audio to Groq Whisper API (language={language})...")

        with open(upload_path, "rb") as audio_file:
            response = requests.post(
                GROQ_TRANSCRIPTION_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
                files={
                    "file": (os.path.basename(upload_path), audio_file),
                },
                data={
                    "model": "whisper-large-v3-turbo",
                    "response_format": "verbose_json",
                    "language": language,
                    "temperature": 0.0,
                },
                timeout=120,
            )

        if response.status_code != 200:
            error_msg = response.text[:500]
            raise ProcessingError(
                f"Groq Whisper API error (HTTP {response.status_code}):\n{error_msg}"
            )

        result = response.json()

        # Step 3: Convert to SRT format
        segments = result.get("segments", [])
        if not segments:
            # Try to get text at least
            text = result.get("text", "")
            if text:
                logger.warning("API returned text but no segments. Creating single-block SRT.")
                duration = result.get("duration", 10.0)
                srt_content = f"1\n00:00:00,000 --> {_format_timestamp(duration)}\n{text}\n\n"
            else:
                raise ProcessingError(
                    "Groq Whisper returned no transcription.\n"
                    "The audio may be too short, silent, or in an unsupported format."
                )
        else:
            srt_content = _segments_to_srt(segments)

        # Step 4: Write SRT file
        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        logger.info(f"SRT file generated: {output_srt_path} ({len(segments)} segments)")
        return output_srt_path

    finally:
        # Clean temp audio files
        if wav_path:
            try:
                import shutil
                shutil.rmtree(os.path.dirname(wav_path), ignore_errors=True)
            except Exception:
                pass


def _compress_audio(input_wav: str, output_mp3: str):
    """Compress WAV to MP3 to fit within Groq's 25MB limit."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ProcessingError("FFmpeg not found — needed to compress audio.")

    cmd = [
        ffmpeg, "-y",
        "-i", input_wav,
        "-codec:a", "libmp3lame",
        "-b:a", "64k",  # Low bitrate to keep size small
        "-ar", "16000",
        "-ac", "1",
        output_mp3,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    if result.returncode != 0:
        raise ProcessingError(f"Audio compression failed:\n{result.stderr[:500]}")


def _segments_to_srt(segments: list) -> str:
    """Convert Whisper API segments to SRT format."""
    srt_lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        if not text:
            continue
        srt_lines.append(
            f"{i}\n"
            f"{_format_timestamp(start)} --> {_format_timestamp(end)}\n"
            f"{text}\n"
        )
    return "\n".join(srt_lines)


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def validate_srt_file(srt_path: str) -> bool:
    """
    Basic validation of an SRT file.

    Returns True if the file looks like valid SRT content.
    """
    if not os.path.isfile(srt_path):
        return False

    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read(2000)  # Check first 2KB

        if not content.strip():
            return False

        # SRT should contain at least one timestamp pattern
        import re
        has_timestamp = re.search(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}", content)
        return has_timestamp is not None

    except Exception:
        return False
