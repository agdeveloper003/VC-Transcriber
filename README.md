# VC Transcriber

**Fully local** transcription of recorded lectures with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and the **Whisper large-v3** model (the most accurate of the Whisper family). An internet connection is only needed on the first run, to download the model (~3 GB).

## How to use

1. Put the recordings in the **`PUT AUDIO FILE HERE`** folder.
2. Double-click **`transcribe.bat`** (on the first run it installs everything by itself).
3. If there are several files, choose whether to transcribe them all in a row or to confirm each file. If you decline a file, you can skip it or quit the process.
4. Type an optional **context prompt** (see below), or press Enter to skip it.
5. The transcripts are saved in **`TRANSCRIPTIONS`**, with the same name as the recording: `lecture2.mp3` → `lecture2.txt`.

From a terminal:

```bash
python transcribe.py
```

If a `.venv` folder exists, the script automatically relaunches itself with the Python inside it, where the CUDA libraries for the GPU are installed.

## Accepted files

- **Audio:** mp3, wav, m4a, aac, flac, ogg, opus, wma, amr, aiff
- **Video:** mp4, mkv, avi, mov, webm, flv, wmv, m4v, mpg, mpeg, ts, mts, m2ts, 3gp, ogv. Only the audio track is read, directly in memory, with no need to convert the file first.

Other files (PDFs, images, videos without audio, ...) are discarded and reported.

## Context prompt

The prompt tells the model what the lecture is about, so it spells technical terms, acronyms and names correctly. For example, without a prompt "Dijkstra" may come out as "DigXtra".

- **Transcribe all in a row:** one prompt is asked once and used for every file.
- **Confirm each file:** a prompt is asked for each file, so every lecture can have its own topic.

A good prompt is a short sentence with the topic followed by the key terms:

```
Computer Networks lecture: TCP, UDP, three-way handshake, Dijkstra, OSPF, BGP.
```

The prompt is applied to the whole recording, not only to its beginning. Keep it under about 800 characters, because Whisper ignores the rest.

## Settings

At the top of `transcribe.py`:

| Constant | Meaning | Default |
|---|---|---|
| `MODEL` | Whisper model (`large-v3`, `large-v3-turbo`, `medium`, ...) | `large-v3` |
| `LANGUAGE` | spoken language (`"en"`, `"it"`, ... or `None` = automatic) | `"en"` |
| `BEAM_SIZE` | search width (higher = slower, little gain above 5) | `5` |
| `PARAGRAPH_GAP` | pause in seconds that starts a new paragraph | `2.0` |

If a `.txt` with the same name already exists in `TRANSCRIPTIONS`, the program asks whether to overwrite it.

## GPU

With an NVIDIA GPU, CUDA is used automatically (the model is tested as soon as it is loaded). If the GPU is not usable, the program explains how to fix it and asks whether to continue on the CPU, which is much slower.
