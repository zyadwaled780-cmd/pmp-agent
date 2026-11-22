@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Local virtual environment not found at "%~dp0.venv".
    echo Run setup_windows.bat first to create .venv and install dependencies.
    pause
    exit /b 1
)

set "PMP_PROVIDER=offline"
if not defined GRADIO_SERVER_NAME set "GRADIO_SERVER_NAME=127.0.0.1"
if not defined PORT set "PORT=7860"
if not defined PMP_OPEN_BROWSER set "PMP_OPEN_BROWSER=1"

echo [PMP Agent] Starting in forced OFFLINE demo mode ^(no external API requests^)
echo [PMP Agent] Local URL: http://%GRADIO_SERVER_NAME%:%PORT%
echo [PMP Agent] Press Ctrl+C in this window to stop the server.

"%~dp0.venv\Scripts\python.exe" "%~dp0app.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo [PMP Agent] App exited with code %RC%.
    pause
)
exit /b %RC%
