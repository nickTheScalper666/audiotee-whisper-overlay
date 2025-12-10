# audiotee-whisper-overlay
System audio capture, live transcription/translation, episode audio recording, and SRT generation powered by AudioTee + Whisper.cpp. Subtitle mp4 video overlay generation


# Whisper Metal Overlay

Desktop overlay for **live subtitles and offline episode subtitles** on macOS, powered by:

- [`whisper.cpp`](https://github.com/ggerganov/whisper.cpp) with **Metal** acceleration  
- [`audiotee`](https://github.com/your/audiotee) (or equivalent) for **system audio capture**  
- A small **Tkinter GUI** for overlay controls

You can:

- See **live subtitles** for any audio playing on your Mac (YouTube, Netflix, etc.).
- Choose the **spoken language** and whether to **translate to English** or stay in the original language.
- Adjust subtitle **font size, weight, and colors** in real time.
- **Record** an entire episode’s system audio to WAV.
- After you stop recording, automatically:
  - Generate an **SRT subtitle file** for the full episode.
  - Generate a **subtitle-only MP4 video** (black background, no audio) with the subtitles burned in.

---

## How it works

### Live overlay

1. `audiotee` captures system audio at 16 kHz and streams raw PCM to stdout.  
2. The Python app reads that stream in small chunks, maintains a sliding window, and periodically:
   - Writes a short segment to a temporary WAV file.
   - Calls `whisper-cli` (from `whisper.cpp`) on that segment.
3. The returned text is shown in a top-most Tkinter window, giving you a “soft subtitle overlay” over your browser / player.

### Record & Segment (offline mode)

When you click **Record**:

1. The app records system audio to `recordings/<episode>.wav`.
2. When you click **Stop Rec**:
   - The recorder stops and closes the WAV file.
   - The app calls `whisper-cli` once on the full WAV to generate an SRT file:
     - Uses the **source language** from the language dropdown (or auto-detect).
     - Uses **translation to English** or **original language**, depending on the output-mode dropdown.
   - It then calls `ffmpeg` to:
     - Create a black 1280×720 video of the correct duration.
     - Burn in the subtitles, producing `videos/<episode>[_en].mp4`.

This MP4 has **no audio** — it is basically a “video transcript” that you can scrub through or keep synced with the original episode.

---

## Features

- **Live subtitles overlay**
  - Top-most Tkinter window, so it sits over YouTube, VLC, etc.
  - Continuous transcription of system audio.
  - De-duping so repeated segments don’t spam the overlay.

- **Language control**
  - Source language dropdown:
    - `Auto-detect`
    - Japanese, Korean, Chinese, Spanish, English…
  - Output mode dropdown:
    - `English (translated)` – whisper.cpp runs with `-tr`  
    - `Original language` – whisper.cpp outputs text in the spoken language

- **Text appearance**
  - Increase / decrease font size (`A-`, `A+`)
  - Toggle bold (`B`)
  - Pick text color
  - Pick background color

- **Recording & full-episode subtitles**
  - Record system audio to `.wav`.
  - Generate `.srt` subtitles for the full file.
  - Generate **subtitle-only MP4** via `ffmpeg` (black background, burned-in subtitles).

- **Anti-garbage cleaning**
  - Removes common hallucinated phrases like
    - “Subtitles by the Amara.org community”
    - “Thanks for watching!”
    - Generic “*music*” lines

---

## Requirements

- **macOS** (tested with AudioTee and whisper.cpp; other platforms would need different system-audio capture)
- **Python 3.9+** with:
  - `numpy`
  - `tk` / Tkinter (usually included with Python on macOS)
- **audiotee** binary available in `PATH` or referenced by `AUDIO_TEE_PATH`
- **whisper.cpp** built with Metal support:
  - `whisper-cli` binary
  - A `.bin` model file (e.g. `ggml-large-v3.bin`)
- **ffmpeg** (for subtitle-only MP4 creation), available in `PATH`

---

## Installation

1. **Clone this repository**

cd ~/somewhere
git clone https://github.com/nickTheScalper666/audiotee-whisper-overlay.git
cd audiotee-whisper-overlay

  
2. Install required tools (Homebrew)  
```bash
brew install cmake pkg-config ffmpeg

3. Get whisper.cpp
Choose a folder (Desktop used as example):
cd ~/Desktop  # or anywhere you prefer
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

4.cmake -B build -DGGML_METAL=ON
cmake --build build -j

5. Download a Whisper model (large models recommended)
Stay inside the whisper.cpp folder:
cd ~/Desktop/whisper.cpp
Option A — Non-quantized GGML model
bash ./models/download-ggml-model.sh large-v2
Result:
models/ggml-large-v2.bin
Option B — Quantized GGUF model (recommended for Apple Silicon)
bash ./models/download-gguf-model.sh large-v2-q5_0
