"""
VC Transcriber - local transcription of recorded lectures with faster-whisper.

Put the recordings in the "PUT AUDIO FILE HERE" folder and run:
    python transcribe.py
Transcripts (.txt with the same name as the recording) are saved in "TRANSCRIPTIONS".
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "PUT AUDIO FILE HERE"
OUTPUT_DIR = BASE_DIR / "TRANSCRIPTIONS"

# --- Transcription settings --------------------------------------------------
# large-v3 is the most accurate Whisper model: in float16 it uses ~5 GB of VRAM.
# (large-v3-turbo is faster but less accurate; there are no larger Whisper models.)
MODEL = "large-v3"
LANGUAGE = "en"            # "en", "it", ... or None for automatic detection
BEAM_SIZE = 5
PARAGRAPH_GAP = 2.0        # pause (s) after which a new paragraph starts in the .txt
# The context prompt is typed in the console. Whisper keeps only ~220 tokens of it
# (roughly 800 characters of English text); anything beyond that is cut off.
PROMPT_MAX_CHARS = 800

# Accepted extensions: audio and video (only the audio track is extracted from videos).
MEDIA_EXT = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr", ".aiff", ".aif",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".3gp", ".ogv",
}


class StopProcessing(Exception):
    pass


def register_cuda_dlls() -> None:
    """On Windows, makes the CUDA DLLs installed via pip (nvidia-cublas/cudnn) visible."""
    if os.name != "nt":
        return
    for base in sys.path:
        nvidia_dir = Path(base) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def relaunch_in_venv() -> None:
    """If the script is started with another Python (e.g. `py transcribe.py`), relaunch it
    with the one in .venv, where the CUDA libraries for the GPU are installed too."""
    venv_dir = BASE_DIR / ".venv"
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_python.exists() or Path(sys.prefix).resolve() == venv_dir.resolve():
        return
    import subprocess
    sys.exit(subprocess.call([str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]]))


def has_audio(path: Path) -> bool:
    import av
    try:
        with av.open(str(path)) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return False


def scan_input() -> tuple[list[Path], list[Path]]:
    """Returns (files with audio, discarded files)."""
    accepted, discarded = [], []
    for f in sorted(INPUT_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() in MEDIA_EXT and has_audio(f):
            accepted.append(f)
        else:
            discarded.append(f)
    return accepted, discarded


def ask(question: str, options: dict[str, str]) -> str:
    """Asks until the user answers with one of the keys of `options`."""
    hint = " / ".join(f"[{k}] {v}" for k, v in options.items())
    while True:
        try:
            answer = input(f"{question}\n  {hint}\n  > ").strip("﻿ \t").lower()
        except EOFError:
            raise StopProcessing from None
        if answer in options:
            return answer
        print("  Invalid answer, please try again.")


def ask_prompt(target: str) -> str | None:
    """Asks for an optional context prompt (topic, technical terms, names) in the console."""
    print(f"\nContext prompt for {target}: course topic, technical terms, acronyms, names.\n"
          "  e.g. \"Computer Networks lecture: TCP, UDP, three-way handshake, Dijkstra, OSPF.\"\n"
          "  Press Enter to skip.")
    try:
        text = input("  > ").strip("﻿ \t")
    except EOFError:
        return None
    if len(text) > PROMPT_MAX_CHARS:
        print(f"  [!] The prompt is {len(text)} characters long: only about the first "
              f"{PROMPT_MAX_CHARS} will be used.")
    return text or None


def decode_audio(path: Path, sampling_rate: int = 16000):
    """Extracts the audio track (from a video too) as a mono 16 kHz float32 array, in memory."""
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError("empty audio track")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def fmt_clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_model():
    register_cuda_dlls()
    import ctranslate2
    from faster_whisper import WhisperModel

    if ctranslate2.get_cuda_device_count() > 0:
        print(f"[*] Loading '{MODEL}' on the GPU (float16)... "
              "(downloaded on first run, ~3 GB)")
        try:
            model = WhisperModel(MODEL, device="cuda", compute_type="float16")
            # Test the GPU right away: the CUDA DLLs (cuBLAS/cuDNN) are only loaded here.
            import numpy as np
            segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32),
                                           language="en", vad_filter=False)
            list(segments)
            return model
        except Exception as e:
            print(f"\n[!] GPU not usable: {e}")
            print("    Fix: run transcribe.bat (or .venv\\Scripts\\python transcribe.py),\n"
                  "    or install into the Python in use: "
                  "pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12==9.*\"")
            if ask("Continue on the CPU? (much slower: hours for a long lecture)",
                   {"y": "yes, use the CPU", "q": "quit"}) == "q":
                raise StopProcessing
    print(f"[*] Loading '{MODEL}' on the CPU (int8): this will be much slower.")
    return WhisperModel(MODEL, device="cpu", compute_type="int8")


def to_paragraphs(segments: list) -> str:
    paragraphs, current, last_end = [], [], None
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if current and last_end is not None and seg.start - last_end > PARAGRAPH_GAP:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        last_end = seg.end
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs) + "\n"


def transcribe_file(model, path: Path, out_path: Path, prompt: str | None) -> None:
    from tqdm import tqdm

    t0 = time.time()
    print("    Extracting audio...")
    audio = decode_audio(path)
    # The prompt is passed as `hotwords` rather than `initial_prompt`: initial_prompt only
    # conditions the first 30 s window, while hotwords is repeated in every window,
    # so the context helps for the whole lecture.
    segments_iter, info = model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        hotwords=prompt,
    )
    print(f"    Language: {info.language}, duration: {fmt_clock(info.duration)}")
    if prompt:
        print(f"    Prompt: {prompt}")

    segments = []
    with tqdm(total=round(info.duration, 1), unit="s", desc="    Transcribing",
              bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]") as bar:
        for seg in segments_iter:
            segments.append(seg)
            bar.update(min(seg.end, info.duration) - bar.n)
        bar.update(bar.total - bar.n)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path.write_text(to_paragraphs(segments), encoding="utf-8")
    elapsed = time.time() - t0
    print(f"    Done in {fmt_clock(elapsed)} -> {out_path.relative_to(BASE_DIR)}")


def confirm_file(path: Path) -> bool:
    """'Ask every time' mode. True = transcribe, False = skip, StopProcessing = quit."""
    if ask(f"\nTranscribe '{path.name}'?", {"y": "yes", "n": "no"}) == "y":
        return True
    if ask("What do you want to do?", {"s": "skip this file", "q": "quit"}) == "s":
        return False
    raise StopProcessing


def main() -> int:
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    files, discarded = scan_input()
    for f in discarded:
        print(f"[-] Discarded (not an audio/video file with audio): {f.name}")
    if not files:
        print(f"No audio files found in '{INPUT_DIR.name}'. Put them there and run again.")
        return 1

    print(f"\n[*] Found {len(files)} file(s):")
    for i, f in enumerate(files, 1):
        print(f"    {i}. {f.name}")

    model = None
    done, skipped, failed = [], [], []
    try:
        ask_each = False
        if len(files) > 1:
            ask_each = ask("\nHow do you want to proceed?",
                           {"1": "transcribe all in a row",
                            "2": "ask for confirmation before each file"}) == "2"
        # Automatic mode: one prompt for all files. Confirmation mode: one prompt per file.
        shared_prompt = None if ask_each else ask_prompt(
            "all files" if len(files) > 1 else f"'{files[0].name}'")
        for i, f in enumerate(files, 1):
            out_path = OUTPUT_DIR / f"{f.stem}.txt"
            if ask_each and not confirm_file(f):
                skipped.append(f)
                continue
            if out_path.exists() and ask(
                    f"\n'{out_path.name}' already exists in TRANSCRIPTIONS. Overwrite it?",
                    {"y": "yes", "n": "no, skip"}) == "n":
                skipped.append(f)
                continue
            prompt = ask_prompt(f"'{f.name}'") if ask_each else shared_prompt
            if model is None:
                model = load_model()
            print(f"\n[{i}/{len(files)}] {f.name}")
            try:
                transcribe_file(model, f, out_path, prompt)
                done.append(f)
            except Exception as e:
                print(f"[X] Error on {f.name}: {e}")
                failed.append(f)
    except StopProcessing:
        print("\n[!] Process stopped on request.")
    except KeyboardInterrupt:
        print("\n[!] Interrupted by the user.")

    print(f"\n[*] Transcribed: {len(done)}  |  Skipped: {len(skipped)}  |  Errors: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    relaunch_in_venv()
    sys.exit(main())
