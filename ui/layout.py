"""
AI Shorts Editor — Main GUI Layout.

Clean, minimal interface built with CustomTkinter.
Handles all user interactions and delegates processing to the engine.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk

from engine.captions import generate_srt_groq, validate_srt_file
from engine.processor import ProcessingError, probe_video, process_video
from ui.theme import (
    ACCENT, ACCENT_HOVER, BG_PRIMARY, BG_SECONDARY, BG_SURFACE, BORDER,
    CORNER_RADIUS, ERROR, FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_HEADING,
    FONT_SIZE_SMALL, FONT_SIZE_TITLE, PADDING_X, PADDING_Y, SUCCESS,
    TEXT_DISABLED, TEXT_PRIMARY, TEXT_SECONDARY, TINT_COLORS, WARNING,
    WINDOW_HEIGHT, WINDOW_WIDTH,
)

logger = logging.getLogger("ShortsEditor.UI")


class ShortsEditorApp(ctk.CTk):
    """Main application window."""

    def __init__(self, project_dir: str):
        super().__init__()

        self.project_dir = project_dir
        self.output_dir = os.path.join(project_dir, "output")
        self.temp_dir = os.path.join(project_dir, "temp")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # State
        self.video_path = None
        self.video_info = None
        self.music_path = None
        self.srt_path = None
        self.is_processing = False

        # Load API key
        self.groq_api_key = os.environ.get("GROQ_API_KEY", "")

        # Window setup
        self.title("AI Shorts Editor")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(480, 700)
        self.configure(fg_color=BG_PRIMARY)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._build_ui()

    def _build_ui(self):
        """Build the complete UI layout."""

        # Scrollable main frame
        self.main_frame = ctk.CTkScrollableFrame(
            self, fg_color=BG_PRIMARY, corner_radius=0
        )
        self.main_frame.pack(fill="both", expand=True, padx=0, pady=0)

        # ---- Title ----
        title_frame = ctk.CTkFrame(self.main_frame, fg_color=BG_SECONDARY,
                                    corner_radius=CORNER_RADIUS)
        title_frame.pack(fill="x", padx=PADDING_X, pady=(PADDING_Y, 5))

        ctk.CTkLabel(
            title_frame, text="🎬  AI Shorts Editor",
            font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
            text_color=TEXT_PRIMARY,
        ).pack(pady=12)

        ctk.CTkLabel(
            title_frame, text="Upload → Configure → Export Shorts-ready video",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        ).pack(pady=(0, 10))

        # ---- Upload Section ----
        self._build_upload_section()

        # ---- Options Section ----
        self._build_options_section()

        # ---- Process Section ----
        self._build_process_section()

    # ------------------------------------------------------------------
    # Upload Section
    # ------------------------------------------------------------------
    def _build_upload_section(self):
        section = ctk.CTkFrame(self.main_frame, fg_color=BG_SECONDARY,
                                corner_radius=CORNER_RADIUS)
        section.pack(fill="x", padx=PADDING_X, pady=PADDING_Y)

        ctk.CTkLabel(
            section, text="📁 Video Input",
            font=(FONT_FAMILY, FONT_SIZE_HEADING, "bold"),
            text_color=TEXT_PRIMARY, anchor="w",
        ).pack(fill="x", padx=15, pady=(12, 5))

        btn_frame = ctk.CTkFrame(section, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=5)

        self.upload_btn = ctk.CTkButton(
            btn_frame, text="Choose Video File",
            command=self._on_upload,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
            height=36, corner_radius=CORNER_RADIUS,
        )
        self.upload_btn.pack(side="left")

        self.file_label = ctk.CTkLabel(
            btn_frame, text="No file selected",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY, anchor="w",
        )
        self.file_label.pack(side="left", padx=10)

        # Video info label
        self.info_label = ctk.CTkLabel(
            section, text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY, anchor="w",
        )
        self.info_label.pack(fill="x", padx=15, pady=(0, 5))

        # Preview frame
        self.preview_frame = ctk.CTkFrame(section, fg_color=BG_PRIMARY,
                                           height=120, corner_radius=6)
        self.preview_frame.pack(fill="x", padx=15, pady=(0, 12))
        self.preview_frame.pack_forget()  # Hidden initially

        self.preview_label = ctk.CTkLabel(
            self.preview_frame, text="", fg_color="transparent",
        )
        self.preview_label.pack(expand=True, padx=5, pady=5)

    # ------------------------------------------------------------------
    # Options Section
    # ------------------------------------------------------------------
    def _build_options_section(self):
        section = ctk.CTkFrame(self.main_frame, fg_color=BG_SECONDARY,
                                corner_radius=CORNER_RADIUS)
        section.pack(fill="x", padx=PADDING_X, pady=PADDING_Y)

        ctk.CTkLabel(
            section, text="⚙️ Editing Options",
            font=(FONT_FAMILY, FONT_SIZE_HEADING, "bold"),
            text_color=TEXT_PRIMARY, anchor="w",
        ).pack(fill="x", padx=15, pady=(12, 8))

        opts_frame = ctk.CTkFrame(section, fg_color="transparent")
        opts_frame.pack(fill="x", padx=15)

        # -- Crop to Shorts --
        self.crop_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opts_frame, text="Crop to Shorts (9:16)",
            variable=self.crop_var, font=(FONT_FAMILY, FONT_SIZE_BODY),
            text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
        ).pack(anchor="w", pady=3)

        # Crop position slider
        self.crop_slider_frame = ctk.CTkFrame(opts_frame, fg_color="transparent")
        self.crop_slider_frame.pack(fill="x", padx=20, pady=(0, 3))

        ctk.CTkLabel(
            self.crop_slider_frame, text="Crop Position:",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        ).pack(side="left")

        self.crop_slider = ctk.CTkSlider(
            self.crop_slider_frame, from_=0, to=1, number_of_steps=20,
            width=150, height=16,
            fg_color=BG_PRIMARY, progress_color=ACCENT,
            button_color=TEXT_PRIMARY, button_hover_color=ACCENT,
        )
        self.crop_slider.set(0.5)
        self.crop_slider.pack(side="left", padx=8)

        self.crop_pos_label = ctk.CTkLabel(
            self.crop_slider_frame, text="Center",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY, width=50,
        )
        self.crop_pos_label.pack(side="left")
        self.crop_slider.configure(command=self._on_crop_slider)

        # -- Add Tint --
        self.tint_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opts_frame, text="Add Tint Overlay",
            variable=self.tint_var, font=(FONT_FAMILY, FONT_SIZE_BODY),
            text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._toggle_tint,
        ).pack(anchor="w", pady=3)

        self.tint_controls = ctk.CTkFrame(opts_frame, fg_color="transparent")
        self.tint_controls.pack(fill="x", padx=20, pady=(0, 3))

        ctk.CTkLabel(
            self.tint_controls, text="Color:",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        ).pack(side="left")

        self.tint_dropdown = ctk.CTkOptionMenu(
            self.tint_controls, values=list(TINT_COLORS.keys()),
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SURFACE, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, width=130,
        )
        self.tint_dropdown.set("Warm Amber")
        self.tint_dropdown.pack(side="left", padx=8)

        ctk.CTkLabel(
            self.tint_controls, text="Opacity:",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(10, 0))

        self.tint_opacity_slider = ctk.CTkSlider(
            self.tint_controls, from_=0.05, to=0.5, number_of_steps=18,
            width=100, height=16,
            fg_color=BG_PRIMARY, progress_color=ACCENT,
            button_color=TEXT_PRIMARY, button_hover_color=ACCENT,
        )
        self.tint_opacity_slider.set(0.2)
        self.tint_opacity_slider.pack(side="left", padx=5)

        self.tint_controls.pack_forget()  # Hidden initially

        # -- Add Background Music --
        self.music_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opts_frame, text="Add Background Music",
            variable=self.music_var, font=(FONT_FAMILY, FONT_SIZE_BODY),
            text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._toggle_music,
        ).pack(anchor="w", pady=3)

        self.music_controls = ctk.CTkFrame(opts_frame, fg_color="transparent")
        self.music_controls.pack(fill="x", padx=20, pady=(0, 3))

        self.music_btn = ctk.CTkButton(
            self.music_controls, text="Select Music File",
            command=self._on_select_music,
            fg_color=BG_SURFACE, hover_color=ACCENT,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            height=28, corner_radius=6,
        )
        self.music_btn.pack(side="left")

        self.music_label = ctk.CTkLabel(
            self.music_controls, text="No file selected",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        self.music_label.pack(side="left", padx=8)

        self.music_controls.pack_forget()  # Hidden initially

        # -- Add Channel Name --
        self.watermark_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            opts_frame, text="Add Channel Name Watermark",
            variable=self.watermark_var, font=(FONT_FAMILY, FONT_SIZE_BODY),
            text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._toggle_watermark,
        ).pack(anchor="w", pady=3)

        self.watermark_controls = ctk.CTkFrame(opts_frame, fg_color="transparent")
        self.watermark_controls.pack(fill="x", padx=20, pady=(0, 3))

        ctk.CTkLabel(
            self.watermark_controls, text="Channel:",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        ).pack(side="left")

        self.channel_entry = ctk.CTkEntry(
            self.watermark_controls, placeholder_text="Your Channel Name",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_PRIMARY, border_color=BORDER,
            text_color=TEXT_PRIMARY, width=200, height=28,
        )
        self.channel_entry.pack(side="left", padx=8)

        # -- Add Captions --
        self.captions_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opts_frame, text="Add Captions",
            variable=self.captions_var, font=(FONT_FAMILY, FONT_SIZE_BODY),
            text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            command=self._toggle_captions,
        ).pack(anchor="w", pady=3)

        self.captions_controls = ctk.CTkFrame(opts_frame, fg_color="transparent")
        self.captions_controls.pack(fill="x", padx=20, pady=(0, 8))

        self.caption_mode_var = ctk.StringVar(value="auto")

        ctk.CTkRadioButton(
            self.captions_controls, text="Auto-generate (Groq Whisper)",
            variable=self.caption_mode_var, value="auto",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
        ).pack(anchor="w", pady=1)

        custom_row = ctk.CTkFrame(self.captions_controls, fg_color="transparent")
        custom_row.pack(fill="x", pady=1)

        ctk.CTkRadioButton(
            custom_row, text="Custom SRT file",
            variable=self.caption_mode_var, value="custom",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
        ).pack(side="left")

        self.srt_btn = ctk.CTkButton(
            custom_row, text="Browse",
            command=self._on_select_srt,
            fg_color=BG_SURFACE, hover_color=ACCENT,
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            height=24, width=70, corner_radius=6,
        )
        self.srt_btn.pack(side="left", padx=8)

        self.srt_label = ctk.CTkLabel(
            custom_row, text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        self.srt_label.pack(side="left")

        # Language selector for auto-captions
        lang_row = ctk.CTkFrame(self.captions_controls, fg_color="transparent")
        lang_row.pack(fill="x", pady=1)

        ctk.CTkLabel(
            lang_row, text="Language:",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(18, 0))

        self.lang_dropdown = ctk.CTkOptionMenu(
            lang_row,
            values=["Hindi (hi)", "English (en)", "Urdu (ur)", "Punjabi (pa)",
                    "Tamil (ta)", "Telugu (te)", "Bengali (bn)"],
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=BG_SURFACE, button_color=ACCENT,
            button_hover_color=ACCENT_HOVER, width=130,
        )
        self.lang_dropdown.set("Hindi (hi)")
        self.lang_dropdown.pack(side="left", padx=8)

        self.captions_controls.pack_forget()  # Hidden initially

        # Bottom padding
        ctk.CTkFrame(section, fg_color="transparent", height=5).pack()

    # ------------------------------------------------------------------
    # Process Section
    # ------------------------------------------------------------------
    def _build_process_section(self):
        section = ctk.CTkFrame(self.main_frame, fg_color=BG_SECONDARY,
                                corner_radius=CORNER_RADIUS)
        section.pack(fill="x", padx=PADDING_X, pady=PADDING_Y)

        self.process_btn = ctk.CTkButton(
            section, text="▶  Process Video",
            command=self._on_process,
            fg_color=ACCENT, hover_color=ACCENT_HOVER,
            font=(FONT_FAMILY, FONT_SIZE_HEADING, "bold"),
            height=42, corner_radius=CORNER_RADIUS,
        )
        self.process_btn.pack(fill="x", padx=15, pady=(12, 8))

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            section, fg_color=BG_PRIMARY, progress_color=ACCENT,
            height=8, corner_radius=4,
        )
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=15, pady=(0, 3))

        self.status_label = ctk.CTkLabel(
            section, text="Ready",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        self.status_label.pack(padx=15, pady=(0, 5))

        # Output section
        self.output_frame = ctk.CTkFrame(section, fg_color="transparent")
        self.output_frame.pack(fill="x", padx=15, pady=(0, 12))

        self.output_btn = ctk.CTkButton(
            self.output_frame, text="📂 Open Output Folder",
            command=self._on_open_output,
            fg_color=BG_SURFACE, hover_color=ACCENT,
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            height=32, corner_radius=6,
        )
        self.output_btn.pack(side="left")

        self.output_label = ctk.CTkLabel(
            self.output_frame, text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=SUCCESS,
        )
        self.output_label.pack(side="left", padx=10)

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------
    def _on_upload(self):
        """Handle video file selection."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"),
                ("MP4", "*.mp4"),
                ("All files", "*.*"),
            ]
        )
        if not file_path:
            return

        self.file_label.configure(text=os.path.basename(file_path))
        self.video_path = file_path

        # Probe video info
        try:
            self.video_info = probe_video(file_path)
            w, h = self.video_info["width"], self.video_info["height"]
            dur = self.video_info["duration"]
            audio = "Yes" if self.video_info["has_audio"] else "No"
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            self.info_label.configure(
                text=f"{w}×{h}  •  {dur:.1f}s  •  Audio: {audio}  •  {size_mb:.1f} MB",
                text_color=TEXT_SECONDARY,
            )

            # Generate preview thumbnail
            self._generate_preview(file_path)

        except ProcessingError as e:
            self.info_label.configure(text=f"⚠ {e}", text_color=WARNING)
            self.video_info = None
        except Exception as e:
            self.info_label.configure(text=f"⚠ Error reading file: {e}", text_color=WARNING)
            self.video_info = None

    def _generate_preview(self, video_path: str):
        """Extract a frame from the video as a thumbnail preview."""
        try:
            import shutil
            import subprocess

            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                return

            preview_path = os.path.join(self.temp_dir, "preview.jpg")
            cmd = [
                ffmpeg, "-y",
                "-i", video_path,
                "-vframes", "1",
                "-ss", "1",  # 1 second in
                "-q:v", "5",
                preview_path,
            ]

            result = subprocess.run(
                cmd, capture_output=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            if result.returncode == 0 and os.path.isfile(preview_path):
                img = Image.open(preview_path)
                # Resize to fit preview area while maintaining aspect ratio
                img.thumbnail((460, 120), Image.Resampling.LANCZOS)
                photo = ctk.CTkImage(light_image=img, dark_image=img,
                                      size=(img.width, img.height))
                self.preview_label.configure(image=photo, text="")
                self.preview_label._image = photo  # Keep reference
                self.preview_frame.pack(fill="x", padx=15, pady=(0, 12))

        except Exception as e:
            logger.warning(f"Preview generation failed: {e}")

    def _on_select_music(self):
        """Handle music file selection."""
        file_path = filedialog.askopenfilename(
            title="Select Music File",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.aac *.m4a *.ogg *.flac"),
                ("MP3", "*.mp3"),
                ("All files", "*.*"),
            ]
        )
        if file_path:
            self.music_path = file_path
            self.music_label.configure(text=os.path.basename(file_path))

    def _on_select_srt(self):
        """Handle custom SRT file selection."""
        file_path = filedialog.askopenfilename(
            title="Select SRT Subtitle File",
            filetypes=[
                ("SRT files", "*.srt"),
                ("All files", "*.*"),
            ]
        )
        if file_path:
            if validate_srt_file(file_path):
                self.srt_path = file_path
                self.srt_label.configure(
                    text=os.path.basename(file_path),
                    text_color=SUCCESS,
                )
            else:
                self.srt_label.configure(
                    text="Invalid SRT file!",
                    text_color=ERROR,
                )
                self.srt_path = None

    def _on_crop_slider(self, value):
        """Update crop position label."""
        if value < 0.3:
            label = "Top"
        elif value > 0.7:
            label = "Bottom"
        else:
            label = "Center"
        self.crop_pos_label.configure(text=label)

    def _toggle_tint(self):
        if self.tint_var.get():
            self.tint_controls.pack(fill="x", padx=20, pady=(0, 3))
        else:
            self.tint_controls.pack_forget()

    def _toggle_music(self):
        if self.music_var.get():
            self.music_controls.pack(fill="x", padx=20, pady=(0, 3))
        else:
            self.music_controls.pack_forget()

    def _toggle_watermark(self):
        if self.watermark_var.get():
            self.watermark_controls.pack(fill="x", padx=20, pady=(0, 3))
        else:
            self.watermark_controls.pack_forget()

    def _toggle_captions(self):
        if self.captions_var.get():
            self.captions_controls.pack(fill="x", padx=20, pady=(0, 8))
        else:
            self.captions_controls.pack_forget()

    def _on_open_output(self):
        """Open the output folder in the system file manager."""
        if os.name == "nt":
            os.startfile(self.output_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", self.output_dir])
        else:
            subprocess.Popen(["xdg-open", self.output_dir])

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def _on_process(self):
        """Validate inputs and start processing in a background thread."""
        if self.is_processing:
            return

        # Validate
        if not self.video_path or not os.path.isfile(self.video_path):
            messagebox.showerror("No Video", "Please select a video file first.")
            return

        if self.music_var.get() and not self.music_path:
            messagebox.showerror("No Music",
                                  "Background music is enabled but no music file selected.")
            return

        if self.watermark_var.get() and not self.channel_entry.get().strip():
            messagebox.showerror("No Channel Name",
                                  "Channel name watermark is enabled but no name entered.")
            return

        if (self.captions_var.get()
                and self.caption_mode_var.get() == "custom"
                and not self.srt_path):
            messagebox.showerror("No SRT File",
                                  "Custom captions enabled but no SRT file selected.")
            return

        # Build output filename
        base_name = Path(self.video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{base_name}_shorts_{timestamp}.mp4"
        output_path = os.path.join(self.output_dir, output_name)

        # Build options dict
        tint_color_name = self.tint_dropdown.get()
        tint_hex = TINT_COLORS.get(tint_color_name, "")

        # Parse language code from dropdown
        lang_text = self.lang_dropdown.get()
        lang_code = lang_text.split("(")[-1].rstrip(")").strip() if "(" in lang_text else "hi"

        options = {
            "crop": self.crop_var.get(),
            "crop_position": self.crop_slider.get(),
            "tint": self.tint_var.get() and bool(tint_hex),
            "tint_color": tint_hex,
            "tint_opacity": self.tint_opacity_slider.get(),
            "music": self.music_var.get(),
            "music_path": self.music_path,
            "music_volume": 0.2,
            "watermark": self.watermark_var.get(),
            "channel_name": self.channel_entry.get().strip(),
            "captions": self.captions_var.get(),
            "srt_path": None,  # Will be set during processing
        }

        # Start processing thread
        self.is_processing = True
        self.process_btn.configure(state="disabled", text="Processing...")
        self.upload_btn.configure(state="disabled")
        self.progress_bar.set(0)
        self.status_label.configure(text="Starting...", text_color=TEXT_SECONDARY)
        self.output_label.configure(text="")

        thread = threading.Thread(
            target=self._process_worker,
            args=(self.video_path, output_path, options, lang_code),
            daemon=True,
        )
        thread.start()

    def _process_worker(self, input_path, output_path, options, lang_code):
        """Background worker for video processing."""
        try:
            # Step 1: Handle captions (if auto-generate)
            if options["captions"]:
                if self.caption_mode_var.get() == "auto":
                    self._update_progress(2, "Generating captions via Groq Whisper...")
                    srt_output = os.path.join(self.temp_dir, "captions.srt")
                    try:
                        generate_srt_groq(
                            input_path, srt_output, self.groq_api_key,
                            language=lang_code,
                        )
                        options["srt_path"] = srt_output
                        self._update_progress(10, "Captions generated!")
                    except ProcessingError as e:
                        # Show warning but continue without captions
                        self._update_progress(10, f"Caption error: {e}")
                        logger.warning(f"Caption generation failed: {e}")
                        options["captions"] = False
                        # Ask user if they want to continue
                        self.after(0, lambda: messagebox.showwarning(
                            "Captions Failed",
                            f"Auto-captions could not be generated:\n\n{e}\n\n"
                            "Processing will continue without captions."
                        ))
                elif self.caption_mode_var.get() == "custom":
                    options["srt_path"] = self.srt_path

            # Step 2: Process video
            process_video(
                input_path=input_path,
                output_path=output_path,
                options=options,
                progress_callback=self._update_progress,
                temp_dir=self.temp_dir,
            )

            # Success
            self.after(0, self._on_process_complete, output_path)

        except ProcessingError as e:
            logger.error(f"Processing failed: {e}")
            self.after(0, self._on_process_error, str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self.after(0, self._on_process_error, f"Unexpected error:\n{e}")

    def _update_progress(self, percent, status):
        """Thread-safe progress update."""
        self.after(0, self._set_progress, percent, status)

    def _set_progress(self, percent, status):
        """Update progress bar and status label (must be called from main thread)."""
        self.progress_bar.set(percent / 100)
        self.status_label.configure(text=status, text_color=TEXT_SECONDARY)

    def _on_process_complete(self, output_path):
        """Called on successful processing."""
        self.is_processing = False
        self.process_btn.configure(state="normal", text="▶  Process Video")
        self.upload_btn.configure(state="normal")
        self.progress_bar.set(1)
        self.status_label.configure(text="✅ Complete!", text_color=SUCCESS)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        self.output_label.configure(
            text=f"{os.path.basename(output_path)} ({size_mb:.1f} MB)"
        )

    def _on_process_error(self, error_msg):
        """Called on processing failure."""
        self.is_processing = False
        self.process_btn.configure(state="normal", text="▶  Process Video")
        self.upload_btn.configure(state="normal")
        self.progress_bar.set(0)
        self.status_label.configure(text="❌ Failed", text_color=ERROR)

        messagebox.showerror("Processing Error", error_msg)
