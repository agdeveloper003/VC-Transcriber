"""
VC Transcriber - trascrizione locale di videolezioni con faster-whisper.

Metti le registrazioni nella cartella "PUT AUDIO FILE HERE" e avvia:
    python transcribe.py
Le trascrizioni (.txt con lo stesso nome della registrazione) finiscono in "TRANSCRIPTIONS".
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "PUT AUDIO FILE HERE"
OUTPUT_DIR = BASE_DIR / "TRANSCRIPTIONS"

# --- Impostazioni di trascrizione -------------------------------------------
# large-v3 è il modello Whisper più accurato: in float16 occupa ~5 GB di VRAM.
# (large-v3-turbo è più veloce ma meno preciso; non esistono Whisper più grandi.)
MODEL = "large-v3"
LANGUAGE = "it"            # "it", "en", ... oppure None per il riconoscimento automatico
BEAM_SIZE = 5
PARAGRAPH_GAP = 2.0        # pausa (s) oltre la quale nel .txt si va a capo
INITIAL_PROMPT = None      # es. "TCP, UDP, handshake, Dijkstra" per aiutare con i termini tecnici

# Estensioni accettate: audio e video (dai video viene estratta solo la traccia audio).
MEDIA_EXT = {
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".amr", ".aiff", ".aif",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".3gp", ".ogv",
}


class StopProcessing(Exception):
    pass


def register_cuda_dlls() -> None:
    """Su Windows rende visibili le DLL CUDA installate via pip (nvidia-cublas/cudnn)."""
    if os.name != "nt":
        return
    for base in sys.path:
        nvidia_dir = Path(base) / "nvidia"
        if not nvidia_dir.is_dir():
            continue
        for bin_dir in nvidia_dir.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def has_audio(path: Path) -> bool:
    import av
    try:
        with av.open(str(path)) as container:
            return len(container.streams.audio) > 0
    except Exception:
        return False


def scan_input() -> tuple[list[Path], list[Path]]:
    """Restituisce (file con audio, file scartati)."""
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
    """Chiede finché l'utente non risponde con una delle chiavi di `options`."""
    hint = " / ".join(f"[{k}] {v}" for k, v in options.items())
    while True:
        try:
            answer = input(f"{question}\n  {hint}\n  > ").strip("﻿ \t").lower()
        except EOFError:
            raise StopProcessing from None
        if answer in options:
            return answer
        print("  Risposta non valida, riprova.")


def decode_audio(path: Path, sampling_rate: int = 16000):
    """Estrae la traccia audio (anche da un video) come array float32 mono 16 kHz, in memoria."""
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
        raise ValueError("traccia audio vuota")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def fmt_clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def load_model():
    register_cuda_dlls()
    import ctranslate2
    from faster_whisper import WhisperModel

    if ctranslate2.get_cuda_device_count() > 0:
        print(f"[*] Carico '{MODEL}' sulla GPU (float16)... "
              "(al primo avvio viene scaricato, ~3 GB)")
        try:
            return WhisperModel(MODEL, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"[!] GPU non utilizzabile ({e}). Ripiego sulla CPU.")
    print(f"[*] Carico '{MODEL}' sulla CPU (int8): sarà molto più lento.")
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


def transcribe_file(model, path: Path, out_path: Path) -> None:
    from tqdm import tqdm

    t0 = time.time()
    print("    Estrazione audio...")
    audio = decode_audio(path)
    segments_iter, info = model.transcribe(
        audio,
        language=LANGUAGE,
        beam_size=BEAM_SIZE,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=INITIAL_PROMPT,
    )
    print(f"    Lingua: {info.language}, durata: {fmt_clock(info.duration)}")

    segments = []
    with tqdm(total=round(info.duration, 1), unit="s", desc="    Trascrizione",
              bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]") as bar:
        for seg in segments_iter:
            segments.append(seg)
            bar.update(min(seg.end, info.duration) - bar.n)
        bar.update(bar.total - bar.n)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path.write_text(to_paragraphs(segments), encoding="utf-8")
    elapsed = time.time() - t0
    print(f"    Fatto in {fmt_clock(elapsed)} -> {out_path.relative_to(BASE_DIR)}")


def confirm_file(path: Path) -> bool:
    """Modalità 'chiedi ogni volta'. True = trascrivi, False = salta, StopProcessing = termina."""
    if ask(f"\nTrascrivere '{path.name}'?", {"s": "sì", "n": "no"}) == "s":
        return True
    if ask("Cosa vuoi fare?", {"s": "salta questo file", "t": "termina"}) == "s":
        return False
    raise StopProcessing


def main() -> int:
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    files, discarded = scan_input()
    for f in discarded:
        print(f"[-] Scartato (non è un file audio/video con audio): {f.name}")
    if not files:
        print(f"Nessun file audio trovato in '{INPUT_DIR.name}'. Inseriscili lì e riavvia.")
        return 1

    print(f"\n[*] Trovati {len(files)} file:")
    for i, f in enumerate(files, 1):
        print(f"    {i}. {f.name}")

    model = None
    done, skipped, failed = [], [], []
    try:
        ask_each = False
        if len(files) > 1:
            ask_each = ask("\nCome vuoi procedere?",
                           {"1": "trascrivi tutti di seguito",
                            "2": "chiedi conferma prima di ogni file"}) == "2"
        for i, f in enumerate(files, 1):
            out_path = OUTPUT_DIR / f"{f.stem}.txt"
            if ask_each and not confirm_file(f):
                skipped.append(f)
                continue
            if out_path.exists() and ask(
                    f"\n'{out_path.name}' esiste già in TRANSCRIPTIONS. Sovrascriverlo?",
                    {"s": "sì", "n": "no, salta"}) == "n":
                skipped.append(f)
                continue
            if model is None:
                model = load_model()
            print(f"\n[{i}/{len(files)}] {f.name}")
            try:
                transcribe_file(model, f, out_path)
                done.append(f)
            except Exception as e:
                print(f"[X] Errore su {f.name}: {e}")
                failed.append(f)
    except StopProcessing:
        print("\n[!] Processo terminato su richiesta.")
    except KeyboardInterrupt:
        print("\n[!] Interrotto dall'utente.")

    print(f"\n[*] Trascritti: {len(done)}  |  Saltati: {len(skipped)}  |  Errori: {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
