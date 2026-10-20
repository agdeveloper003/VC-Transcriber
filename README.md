# VC Transcriber

Trascrizione **completamente in locale** di videolezioni con [faster-whisper](https://github.com/SYSTRAN/faster-whisper) e il modello **Whisper large-v3** (il più accurato della famiglia Whisper). Internet serve solo al primo avvio, per scaricare il modello (~3 GB).

## Come si usa

1. Metti le registrazioni nella cartella **`PUT AUDIO FILE HERE`**.
2. Fai doppio clic su **`trascrivi.bat`** (al primo avvio installa tutto da solo).
3. Se ci sono più file, scegli se trascriverli tutti di seguito o se dare la conferma file per file. Se rifiuti un file, puoi saltarlo o terminare il processo.
4. Trovi le trascrizioni in **`TRANSCRIPTIONS`**, con lo stesso nome della registrazione: `lezione2.mp3` → `lezione2.txt`.

Da terminale:

```bash
.venv\Scripts\python transcribe.py
```

## File accettati

- **Audio:** mp3, wav, m4a, aac, flac, ogg, opus, wma, amr, aiff
- **Video:** mp4, mkv, avi, mov, webm, flv, wmv, m4v, mpg, mpeg, ts, mts, m2ts, 3gp, ogv. Viene letta solo la traccia audio, direttamente in memoria, senza bisogno di convertire prima il file.

Gli altri file (PDF, immagini, video senza audio, ...) vengono scartati e segnalati.

## Impostazioni

All'inizio di `transcribe.py`:

| Costante | Significato | Default |
|---|---|---|
| `MODEL` | modello Whisper (`large-v3`, `large-v3-turbo`, `medium`, ...) | `large-v3` |
| `LANGUAGE` | lingua parlata (`"it"`, `"en"`, ... oppure `None` = automatica) | `"it"` |
| `BEAM_SIZE` | ampiezza della ricerca (più alto = più lento, poco guadagno oltre 5) | `5` |
| `INITIAL_PROMPT` | termini tecnici del corso, per riconoscerli meglio | `None` |
| `PARAGRAPH_GAP` | pausa in secondi che genera un nuovo paragrafo | `2.0` |

Se nella cartella `TRANSCRIPTIONS` esiste già un `.txt` con lo stesso nome, il programma chiede se sovrascriverlo.
