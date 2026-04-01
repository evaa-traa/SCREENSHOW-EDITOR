"""
AI Shorts Editor — Desktop Application
Entry point for the YouTube Shorts video automation tool.
"""

import logging
import os
import sys
from pathlib import Path

# Setup logging before anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ShortsEditor")


def main():
    # Determine project root
    project_dir = str(Path(__file__).parent.resolve())

    # Load environment variables from .env
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(project_dir, ".env")
        if os.path.isfile(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded .env from {env_path}")
        else:
            logger.warning(f".env file not found at {env_path}")
    except ImportError:
        logger.warning("python-dotenv not installed. Set GROQ_API_KEY manually.")

    # Ensure required directories exist
    for dirname in ("output", "temp", "assets/music", "assets/fonts"):
        os.makedirs(os.path.join(project_dir, dirname), exist_ok=True)

    # Check FFmpeg availability early
    import shutil
    if not shutil.which("ffmpeg"):
        logger.error(
            "FFmpeg not found! The app requires FFmpeg to process videos.\n"
            "Download from https://ffmpeg.org/download.html and add to PATH."
        )
        # Still launch the app — it will show error when user tries to process
    else:
        ffmpeg_path = shutil.which("ffmpeg")
        logger.info(f"FFmpeg found: {ffmpeg_path}")

    # Launch GUI
    from ui.layout import ShortsEditorApp

    app = ShortsEditorApp(project_dir=project_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
