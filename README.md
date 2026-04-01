# AI Shorts Editor V1

A desktop application designed to automate YouTube Shorts video editing. Upload a source video, select your desired editing options, and let the app handle the rest — including center-cropping to 9:16, resizing to 1080x1920, generating AI captions, adding background music, applying color overlays, and rendering a channel watermark.

Built with Python, CustomTkinter, FFmpeg, and the Groq Whisper API.

## Features

- **Crop to Shorts (9:16)**: Automatically center-crops landscape videos and scales to 1080x1920 with adjustable vertical framing.
- **AI Captions (Hindi/Multilingual)**: Uses Groq's high-speed `whisper-large-v3-turbo` model to generate and burn subtitles, or load your own `.srt` file.
- **Background Music**: Mixes an audio track gracefully over the original video, auto-looping if necessary.
- **Color Tint Overlay**: Apply cinematic color washes (Warm Amber, Cool Blue, Sepia, etc.) with configurable opacity.
- **Channel Watermark**: Imprints your channel name at the bottom center of the video.
- **Modern Dark UI**: A clean, responsive interface powered by CustomTkinter.

## Prerequisites

1. **Python 3.8+**
2. **FFmpeg & FFprobe**: 
   - This app relies on direct FFmpeg subprocess calls to process video.
   - **Windows**: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (get the "essentials" release). Extract the zip and add the `bin` folder to your Windows system `PATH`.
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`
3. **Groq API Key**: 
   - Get a free API key at [console.groq.com](https://console.groq.com). Required for auto-generating captions.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "AI Shorts Editor"
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   - Copy `.env.example` to a new file named `.env`:
     ```bash
     cp .env.example .env
     ```
   - Open `.env` and add your Groq API key:
     ```env
     GROQ_API_KEY=gsk_your_actual_key_here
     ```

## Usage

1. Launch the application:
   ```bash
   python app.py
   ```
2. Click **Choose Video File** and select your source video.
3. Toggle options like **Crop**, **Tint**, **Music**, **Watermark**, and **Captions**.
4. Click **Process Video**.
5. Once complete, click **Open Output Folder** to retrieve your Shorts-ready video.

## Project Structure

```
.
├── app.py                  # Main entry point; initializes directories and launches GUI
├── engine/                 # Core processing logic
│   ├── processor.py        # Orchestrates the FFmpeg filter complex (crop, tint, mix, subtitle burn)
│   └── captions.py         # Extracts audio, contacts Groq API, formats SRT files
├── ui/                     # Presentation layer
│   ├── layout.py           # CustomTkinter interface, form validation, and background threading
│   └── theme.py            # Styling constants and color presets
├── assets/                 # (Created dynamically) Fonts, music
├── output/                 # (Created dynamically) Processed videos
├── temp/                   # (Created dynamically) Auto-cleaned temporary files
├── requirements.txt        
└── .gitignore             
```

## Technical Notes

- **Robust Processing**: The `engine/processor.py` entirely avoids wrapper libraries (like `ffmpeg-python` or `moviepy`) in favor of direct CLI subprocess calls. This allows exhaustive error capturing, real-time progression parsing (`time=xx:xx:xx`), and prevents silent failures. 
- **Graceful Fallbacks**: If the Groq API fails or is unconfigured, the pipeline will securely skip the subtitle burn-in step rather than crashing the overall video render. Temporary files (extracted WAVs, generated SRTs) are rigorously cleaned post-render.

## License

MIT License (or your chosen license)
