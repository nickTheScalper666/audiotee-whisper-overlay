import os
import re
import glob
import subprocess
import threading
import queue
import time
import tempfile
import wave
from typing import Optional

import numpy as np

import tkinter as tk
from tkinter import scrolledtext, colorchooser, simpledialog, messagebox
from tkinter import ttk
import tkinter.font as tkfont


# -----------------------------
# CONFIG
# -----------------------------

# AudioTee binary (we built this earlier)
AUDIO_TEE_PATH = "audiotee"  # change to full path if which audiotee fails

# whisper.cpp binary built with Metal
WHISPER_BIN = "/Users/nickshroff/whisper.cpp/build/bin/whisper-cli"  # ### CHECK THIS PATH ###
WHISPER_MODEL = "/Users/nickshroff/whisper.cpp/models/ggml-large-v3.bin"  # ### CHECK THIS PATH ###

# Audio params
SAMPLE_RATE = 16000
CHUNK_SEC = 2.0           # AudioTee chunk duration
SEGMENT_SEC = 12.0        # audio length sent to whisper.cpp (seconds)
STEP_SEC = 6.0            # hop size (overlap = SEGMENT_SEC - STEP_SEC)

# Record & segment paths
RECORDINGS_DIR = "recordings"
SUBTITLES_DIR = "subtitles"
VIDEO_DIR = "videos"

# Language dropdown options (source / spoken language)
LANG_DISPLAY_TO_CODE = {
    "Auto-detect": None,
    "Hindi (hi)": "hi",
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
    "Chinese (zh)": "zh",
    "Hebrew (he)": "he",
    "Telugu (te)": "te",
    "Kannada (kn)": "kn",
    "English (en)": "en",
}


class AppConfig:
    def __init__(self) -> None:
        self._output_mode = "english"     # "english" or "original"
        self._source_lang: Optional[str] = None  # e.g. "hi", "ja", or None for auto

    def set_output_mode(self, mode: str) -> None:
        self._output_mode = mode

    def get_output_mode(self) -> str:
        return self._output_mode

    def set_source_lang(self, code: Optional[str]) -> None:
        self._source_lang = code

    def get_source_lang(self) -> Optional[str]:
        return self._source_lang


def which_exec(path: str) -> Optional[str]:
    """Locate an executable in PATH or return absolute path if already absolute."""
    if os.path.isabs(path):
        return path if os.access(path, os.X_OK) else None
    for directory in os.getenv("PATH", "").split(os.pathsep):
        candidate = os.path.join(directory, path)
        if os.access(candidate, os.X_OK):
            return candidate
    return None


def run_whisper_cpp(
    segment: np.ndarray,
    output_mode: str,
    lang_code: Optional[str],
) -> str:
    """
    Run whisper.cpp (Metal) on one audio segment.
    segment: float32 mono [-1, 1] at SAMPLE_RATE
    output_mode: "english" or "original"
    lang_code: None for auto-detect, or e.g. "hi", "ja", "en", etc.
    """
    # 1. write WAV for this segment
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        pcm = (segment * 32767.0).clip(-32768, 32767).astype("<i2")
        wf.writeframes(pcm.tobytes())

    txt_prefix = wav_path
    txt_path = wav_path + ".txt"

    whisper_lang = lang_code if lang_code else "auto"

    cmd = [
        WHISPER_BIN,
        "-m",
        WHISPER_MODEL,
        "-f",
        wav_path,
        "-l",
        whisper_lang,
        "-otxt",
        "-of",
        txt_prefix,
    ]
    if output_mode == "english":
        cmd.append("-tr")  # translate to English

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("[whisper.cpp ERROR]", result.stderr.strip())

    text = ""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except FileNotFoundError:
        text = ""

    # clean up temp files
    for p in (wav_path, txt_path):
        try:
            os.remove(p)
        except OSError:
            pass

    # --- simple anti-hallucination cleanup ---
    if text:
        garbage_phrases = [
            "Subtitles by the Amara.org community",
            "Subtitles by Amara.org",
            "Subtitles by the community",
            "Thanks for watching!",
            "*music*",
            "* Badass music *",
        ]
        for g in garbage_phrases:
            text = text.replace(g, "")
        text = text.strip()

    return text


# -----------------------------
# OFFLINE RECORD & SEGMENT
# -----------------------------

def record_worker(output_wav_path: str, stop_event: threading.Event) -> None:
    """Record raw system audio via audiotee into a WAV file."""
    audiotee_exe = which_exec(AUDIO_TEE_PATH)
    if not audiotee_exe:
        print(f"[RECORD ERROR] audiotee not found at '{AUDIO_TEE_PATH}'.")
        return

    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

    rec_chunk_sec = 0.5  # smaller chunk for more responsive stopping
    bytes_per_sample = 2
    chunk_bytes = int(SAMPLE_RATE * rec_chunk_sec * bytes_per_sample)

    cmd = [
        audiotee_exe,
        "--sample-rate",
        str(SAMPLE_RATE),
        "--chunk-duration",
        str(rec_chunk_sec),
    ]
    print("[RECORD] Starting audiotee:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    try:
        with wave.open(output_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)

            while not stop_event.is_set():
                if proc.stdout is None:
                    print("[RECORD ERROR] audiotee stdout is None")
                    break

                raw = proc.stdout.read(chunk_bytes)
                if not raw:
                    print("[RECORD] audiotee EOF")
                    break

                wf.writeframes(raw)

    finally:
        try:
            proc.terminate()
        except Exception:
            pass
        print(f"[RECORD] Finished. Saved to {output_wav_path}")


def _parse_srt_duration(srt_path: str) -> float:
    """
    Parse an SRT file and return total duration in seconds
    based on the last subtitle's end time.
    """
    time_re = re.compile(
        r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)"
    )
    last_end = 0.0
    with open(srt_path, "r", encoding="utf-8") as f:
        for line in f:
            m = time_re.match(line.strip())
            if not m:
                continue
            h2, m2, s2, ms2 = m.group(5, 6, 7, 8)
            t = (
                int(h2) * 3600
                + int(m2) * 60
                + int(s2)
                + int(ms2) / 1000.0
            )
            if t > last_end:
                last_end = t
    # If for some reason nothing parsed, default to 60s
    if last_end <= 0:
        last_end = 60.0
    # small padding so last line is visible
    return last_end + 1.0


def create_subtitle_video(srt_path: str) -> Optional[str]:
    """
    Use ffmpeg to create a silent MP4 video with subtitles burned in.
    Background = black, no audio, size 1280x720.

    Returns the MP4 path, or None on failure.
    """
    ffmpeg_exe = which_exec("ffmpeg")
    if not ffmpeg_exe:
        print("[VIDEO ERROR] ffmpeg not found in PATH.")
        return None

    if not os.path.isfile(srt_path):
        print("[VIDEO ERROR] SRT not found:", srt_path)
        return None

    duration = _parse_srt_duration(srt_path)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(srt_path))[0]
    mp4_path = os.path.join(VIDEO_DIR, base_name + ".mp4")

    # Escape characters that confuse ffmpeg's subtitles filter
    ff_srt_path = srt_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    vf_expr = (
        f"subtitles={ff_srt_path}:"
        "force_style='Fontsize=32,PrimaryColour=&HFFFFFF&'"
    )

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=1280x720:d={duration}",
        "-vf",
        vf_expr,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        mp4_path,
    ]

    print("[VIDEO] Running ffmpeg to create subtitle video...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[VIDEO ERROR]", result.stderr.strip())
        return None

    print("[VIDEO] Created:", mp4_path)
    return mp4_path


def segment_episode(
    wav_path: str,
    lang_code: Optional[str],
    translate_to_english: bool = True,
) -> Optional[str]:
    """
    Run whisper.cpp on a full episode WAV, producing an SRT file.

    More robust:
      - prints stdout / stderr
      - tries exact SRT name, then any matching prefix (handles variant names)
    """
    if not os.path.isfile(WHISPER_BIN):
        print("[SEGMENT ERROR] whisper.cpp binary not found.")
        return None
    if not os.path.isfile(WHISPER_MODEL):
        print("[SEGMENT ERROR] whisper model not found.")
        return None

    os.makedirs(SUBTITLES_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    suffix = "_en" if translate_to_english else ""
    out_prefix = os.path.join(SUBTITLES_DIR, base_name + suffix)

    whisper_lang = lang_code if lang_code else "auto"

    cmd = [
        WHISPER_BIN,
        "-m",
        WHISPER_MODEL,
        "-f",
        wav_path,
        "-l",
        whisper_lang,
        "-osrt",
        "-of",
        out_prefix,
        "-bs",
        "8",
    ]
    if translate_to_english:
        cmd.append("-tr")

    print("[SEGMENT] Running whisper.cpp for full-episode subtitles...")
    print("[SEGMENT] Command:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print("[SEGMENT stdout]")
        print(result.stdout)
    if result.stderr:
        print("[SEGMENT stderr]")
        print(result.stderr)

    if result.returncode != 0:
        print("[SEGMENT ERROR] whisper.cpp returned non-zero exit code:", result.returncode)
        return None

    # Expected exact path
    srt_path = out_prefix + ".srt"
    if os.path.isfile(srt_path):
        print("[SEGMENT] SRT created:", srt_path)
        return srt_path

    # Fallback: any SRT that starts with the prefix (handles versions that add extra suffix)
    pattern = out_prefix + "*.srt"
    candidates = sorted(glob.glob(pattern))
    if candidates:
        srt_path = candidates[0]
        print(f"[SEGMENT] Exact SRT not found, using fallback: {srt_path}")
        return srt_path

    print("[SEGMENT ERROR] Expected SRT not found for prefix:", out_prefix)
    return None


# -----------------------------
# AUDIO WORKER (AudioTee + whisper.cpp)
# -----------------------------

def audio_worker(
    config: AppConfig,
    text_queue: "queue.Queue[str]",
    stop_event: threading.Event,
) -> None:

    audiotee_exe = which_exec(AUDIO_TEE_PATH)
    if not audiotee_exe:
        print(f"[ERROR] audiotee not found at '{AUDIO_TEE_PATH}'.")
        return

    if not os.path.isfile(WHISPER_BIN):
        print(f"[ERROR] whisper.cpp binary not found at '{WHISPER_BIN}'.")
        return

    if not os.path.isfile(WHISPER_MODEL):
        print(f"[ERROR] whisper model not found at '{WHISPER_MODEL}'.")
        return

    print(f"[INFO] Using audiotee: {audiotee_exe}")
    print(f"[INFO] Using whisper.cpp: {WHISPER_BIN}")
    print(f"[INFO] Model: {WHISPER_MODEL}")

    bytes_per_sample = 2
    chunk_bytes = int(SAMPLE_RATE * CHUNK_SEC * bytes_per_sample)

    segment_samples = int(SEGMENT_SEC * SAMPLE_RATE)
    step_samples = int(STEP_SEC * SAMPLE_RATE)

    cmd = [
        audiotee_exe,
        "--sample-rate",
        str(SAMPLE_RATE),
        "--chunk-duration",
        str(CHUNK_SEC),
    ]
    print("[INFO] Starting audiotee:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )

    buffer = np.zeros(0, dtype=np.float32)
    last_line = ""

    try:
        while not stop_event.is_set():
            if proc.stdout is None:
                print("[ERROR] audiotee stdout is None")
                break

            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                print("[INFO] audiotee EOF")
                break

            pcm_i16 = np.frombuffer(raw, dtype="<i2")
            audio = pcm_i16.astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, audio])

            while buffer.shape[0] >= segment_samples:
                segment = buffer[:segment_samples]

                output_mode = config.get_output_mode()
                lang_code = config.get_source_lang()
                text = run_whisper_cpp(segment, output_mode, lang_code)

                if text and text != last_line:
                    last_line = text
                    text_queue.put(text)

                buffer = buffer[step_samples:]

            time.sleep(0.005)

    finally:
        try:
            proc.terminate()
        except Exception:
            pass


# -----------------------------
# TKINTER GUI
# -----------------------------

def start_gui() -> None:
    config = AppConfig()
    text_queue: "queue.Queue[str]" = queue.Queue()
    stop_event = threading.Event()

    worker_thread = threading.Thread(
        target=audio_worker,
        args=(config, text_queue, stop_event),
        daemon=True,
    )
    worker_thread.start()

    root = tk.Tk()
    root.title("Whisper Metal Overlay")
    root.attributes("-topmost", True)
    root.configure(bg="#202020")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("Toolbar.TFrame", background="#202020")
    style.configure("Toolbar.TButton", padding=4)
    style.configure("Toolbar.TMenubutton", padding=4)

    font_family = "Helvetica"
    font_size = 18
    is_bold = False
    text_color = "#FFFFFF"  # white by default
    bg_color = "#000000"    # black background

    subtitle_font = tkfont.Font(
        family=font_family,
        size=font_size,
        weight="normal",
    )

    toolbar = ttk.Frame(root, style="Toolbar.TFrame")
    toolbar.pack(side=tk.TOP, fill=tk.X)

    # ---- Output language mode (English vs original) ----
    output_options = [
        "English (translated)",
        "Original language",
    ]
    selected_output = tk.StringVar(value=output_options[0])

    def on_output_change(*_args) -> None:
        choice = selected_output.get()
        if choice.startswith("English"):
            config.set_output_mode("english")
        else:
            config.set_output_mode("original")

    output_menu = ttk.OptionMenu(
        toolbar,
        selected_output,
        *output_options,
        command=lambda *_: on_output_change(),
    )
    output_menu.config(style="Toolbar.TMenubutton")
    output_menu.pack(side=tk.LEFT, padx=4, pady=2)

    # ---- Source language selector (spoken language) ----
    lang_display_options = list(LANG_DISPLAY_TO_CODE.keys())
    selected_lang = tk.StringVar(value="Auto-detect")

    def on_lang_change(*_args) -> None:
        display = selected_lang.get()
        code = LANG_DISPLAY_TO_CODE.get(display)
        config.set_source_lang(code)

    lang_menu = ttk.OptionMenu(
        toolbar,
        selected_lang,
        *lang_display_options,
        command=lambda *_: on_lang_change(),
    )
    lang_menu.config(style="Toolbar.TMenubutton")
    lang_menu.pack(side=tk.LEFT, padx=4, pady=2)

    # ---- Font controls ----

    def update_font() -> None:
        nonlocal font_size, is_bold, text_color, bg_color
        weight = "bold" if is_bold else "normal"
        subtitle_font.configure(size=font_size, weight=weight)
        text_box.configure(font=subtitle_font, fg=text_color, bg=bg_color)

    def increase_font() -> None:
        nonlocal font_size
        font_size += 2
        update_font()

    def decrease_font() -> None:
        nonlocal font_size
        font_size = max(8, font_size - 2)
        update_font()

    btn_dec = ttk.Button(
        toolbar,
        text="A−",
        style="Toolbar.TButton",
        command=decrease_font,
    )
    btn_inc = ttk.Button(
        toolbar,
        text="A+",
        style="Toolbar.TButton",
        command=increase_font,
    )
    btn_dec.pack(side=tk.LEFT, padx=2)
    btn_inc.pack(side=tk.LEFT, padx=2)

    def toggle_bold() -> None:
        nonlocal is_bold
        is_bold = not is_bold
        update_font()

    btn_bold = ttk.Button(
        toolbar,
        text="B",
        style="Toolbar.TButton",
        command=toggle_bold,
    )
    btn_bold.pack(side=tk.LEFT, padx=2)

    def pick_text_color() -> None:
        nonlocal text_color
        color = colorchooser.askcolor(
            color=text_color,
            title="Choose text color",
        )[1]
        if color:
            text_color = color
            update_font()

    btn_text_color = ttk.Button(
        toolbar,
        text="Text Color",
        style="Toolbar.TButton",
        command=pick_text_color,
    )
    btn_text_color.pack(side=tk.LEFT, padx=2)

    def pick_bg_color() -> None:
        nonlocal bg_color
        color = colorchooser.askcolor(
            color=bg_color,
            title="Choose background color",
        )[1]
        if color:
            bg_color = color
            update_font()

    btn_bg_color = ttk.Button(
        toolbar,
        text="BG Color",
        style="Toolbar.TButton",
        command=pick_bg_color,
    )
    btn_bg_color.pack(side=tk.LEFT, padx=2)

    # ---- Record & Segment controls ----

    status_var = tk.StringVar(value="Overlay running. Not recording.")
    record_thread: Optional[threading.Thread] = None
    record_stop_event: Optional[threading.Event] = None
    current_recording_path: Optional[str] = None

    def start_recording() -> None:
        nonlocal record_thread, record_stop_event, current_recording_path
        if record_thread is not None and record_thread.is_alive():
            print("[RECORD] Already recording.")
            return

        episode_name = simpledialog.askstring(
            "Record Episode",
            "Episode name / ID (e.g. doraemon_877):",
            parent=root,
        )
        if not episode_name:
            return

        # sanitize: letters, digits, underscore, dash; no spaces (keeps ffmpeg happy)
        safe_name_chars = []
        for c in episode_name:
            if c.isalnum() or c in ("_", "-"):
                safe_name_chars.append(c)
            else:
                safe_name_chars.append("_")
        safe_name = "".join(safe_name_chars).strip("_")
        if not safe_name:
            safe_name = "episode"

        wav_path = os.path.join(RECORDINGS_DIR, safe_name + ".wav")
        record_stop_event = threading.Event()
        current_recording_path = wav_path

        record_thread = threading.Thread(
            target=record_worker,
            args=(wav_path, record_stop_event),
            daemon=True,
        )
        record_thread.start()
        status_var.set(f"Recording: {safe_name}.wav")

    def _finish_recording_and_segment(
        rec_thread: threading.Thread,
        wav_path: str,
        lang_code: Optional[str],
        output_mode: str,
    ) -> None:
        rec_thread.join()
        print("[RECORD] Join complete.")
        translate_to_english = output_mode == "english"

        # 1) Make SRT subtitles
        srt_path = segment_episode(wav_path, lang_code, translate_to_english)

        video_path = None
        if srt_path:
            # 2) Make subtitle-only MP4 video
            video_path = create_subtitle_video(srt_path)

        if srt_path and video_path:
            msg = (
                f"Recording saved:\n{wav_path}\n\n"
                f"Subtitles created:\n{srt_path}\n\n"
                f"Subtitle video created:\n{video_path}"
            )
        elif srt_path:
            msg = (
                f"Recording saved:\n{wav_path}\n\n"
                f"Subtitles created:\n{srt_path}\n\n"
                f"(Video creation failed; see console.)"
            )
        else:
            msg = (
                f"Recording saved:\n{wav_path}\n\n"
                f"(Subtitle generation failed; see console.)"
            )

        try:
            messagebox.showinfo("Record & Segment", msg, parent=root)
        except tk.TclError:
            print("[INFO]", msg)

        status_var.set("Overlay running. Not recording.")

    def stop_recording() -> None:
        nonlocal record_thread, record_stop_event, current_recording_path
        if record_thread is None or not record_thread.is_alive():
            print("[RECORD] No active recording.")
            return

        assert record_stop_event is not None
        assert current_recording_path is not None

        record_stop_event.set()
        status_var.set("Finishing recording and generating subtitles...")

        # Run join + segmentation + video build in helper thread so GUI doesn't freeze
        helper = threading.Thread(
            target=_finish_recording_and_segment,
            args=(
                record_thread,
                current_recording_path,
                config.get_source_lang(),
                config.get_output_mode(),
            ),
            daemon=True,
        )
        helper.start()

        # clear local references
        record_thread = None
        record_stop_event = None
        current_recording_path = None

    btn_rec = ttk.Button(
        toolbar,
        text="Record",
        style="Toolbar.TButton",
        command=start_recording,
    )
    btn_stop_rec = ttk.Button(
        toolbar,
        text="Stop Rec",
        style="Toolbar.TButton",
        command=stop_recording,
    )
    btn_rec.pack(side=tk.LEFT, padx=4)
    btn_stop_rec.pack(side=tk.LEFT, padx=2)

    text_box = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        height=4,
        bg=bg_color,
        fg=text_color,
        font=subtitle_font,
        bd=0,
        highlightthickness=0,
    )
    text_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    status_label = ttk.Label(
        root,
        textvariable=status_var,
        background="#202020",
        foreground="#CCCCCC",
    )
    status_label.pack(side=tk.BOTTOM, anchor="w", padx=4, pady=2)

    root.rowconfigure(1, weight=1)
    root.columnconfigure(0, weight=1)

    update_font()

    def poll_queue() -> None:
        try:
            while True:
                line = text_queue.get_nowait()
                text_box.insert(tk.END, line + "\n")
                text_box.see(tk.END)
        except queue.Empty:
            pass
        root.after(100, poll_queue)

    def on_close() -> None:
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.after(100, poll_queue)
    root.mainloop()


if __name__ == "__main__":
    start_gui()

