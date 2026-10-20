@echo off
REM Doppio clic per trascrivere i file nella cartella "PUT AUDIO FILE HERE".
setlocal
set "HERE=%~dp0"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_IMPLICIT_TOKEN=1"

if not exist "%HERE%.venv\Scripts\python.exe" (
    echo [*] Primo avvio: creo l'ambiente virtuale e installo le dipendenze...
    python -m venv "%HERE%.venv" || goto :error
    "%HERE%.venv\Scripts\python.exe" -m pip install --upgrade pip
    "%HERE%.venv\Scripts\python.exe" -m pip install -r "%HERE%requirements.txt" || goto :error
)

"%HERE%.venv\Scripts\python.exe" "%HERE%transcribe.py"
pause
exit /b

:error
echo [X] Installazione fallita.
pause
exit /b 1
