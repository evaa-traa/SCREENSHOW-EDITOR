# AI Shorts Editor V1

A web application that automates YouTube Shorts video editing. Upload a source video, toggle editing options (crop, captions, music, tint, watermark), and download a ready-to-upload 1080×1920 MP4.

Built with Python, Flask, FFmpeg, and the Groq Whisper API.

## Features

- **Crop to Shorts (9:16)** — Center-crops and scales to 1080×1920 with adjustable vertical framing
- **AI Captions (Hindi/Multilingual)** — Groq `whisper-large-v3-turbo` for auto-generated subtitles, or upload your own `.srt`
- **Background Music** — Mix an audio track at your desired volume, auto-looping if necessary
- **Color Tint Overlay** — Cinematic color washes (Warm Amber, Cool Blue, Sepia, etc.) with configurable opacity
- **Channel Watermark** — Your channel name at the bottom center of the video
- **Web-based UI** — Dark themed, responsive, drag-and-drop uploads

## Prerequisites

- **Python 3.8+**
- **FFmpeg & FFprobe** (must be in PATH)
- **Groq API Key** — Free at [console.groq.com](https://console.groq.com) (required for auto-captions)

## Local Setup

```bash
git clone <repo-url>
cd "SCREENSHOW EDITOR"
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=gsk_your_key_here
```

Run:
```
python app.py
```

Open http://localhost:5000

## Render Deployment

1. Create a new **Web Service** on [Render](https://render.com)
2. Connect your GitHub repo
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Add environment variable: `GROQ_API_KEY` = your key
5. Deploy

> **Note:** Render free tier has limited resources. Short videos (under 60s) work best.

## Project Structure

```
├── app.py                  # Flask web server (routes, upload, status, download)
├── engine/
│   ├── processor.py        # FFmpeg pipeline (crop, tint, music, watermark, subtitles)
│   └── captions.py         # Groq Whisper API + custom SRT support
├── templates/
│   └── index.html          # Web UI (dark theme, responsive)
├── uploads/                # (auto-created) Temporary uploaded files
├── output/                 # (auto-created) Processed videos
├── temp/                   # (auto-created) FFmpeg temp files
├── requirements.txt
└── .env.example
```

## License

MIT
