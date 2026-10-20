@echo off
REM Double-click to transcribe the files in the "PUT AUDIO FILE HERE" folder.
setlocal
set "HERE=%~dp0"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_IMPLICIT_TOKEN=1"

if not exist "%HERE%.venv\Scripts\python.exe" (
    echo [*] First run: creating the virtual environment and installing dependencies...
    python -m venv "%HERE%.venv" || goto :error
    "%HERE%.venv\Scripts\python.exe" -m pip install --upgrade pip
    "%HERE%.venv\Scripts\python.exe" -m pip install -r "%HERE%requirements.txt" || goto :error
)

"%HERE%.venv\Scripts\python.exe" "%HERE%transcribe.py"
pause
exit /b

:error
echo [X] Installation failed.
pause
exit /b 1
