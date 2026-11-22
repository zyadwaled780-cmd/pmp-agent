@echo off
setlocal
cd /d "%~dp0"

set "INSTALL_DEV=0"
if /I "%~1"=="--dev" set "INSTALL_DEV=1"
if /I "%~1"=="-dev" set "INSTALL_DEV=1"

echo [PMP Agent] Checking Python 3.11+...
set "BASE_PY="
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
    if %ERRORLEVEL%==0 set "BASE_PY=py -3"
)

if not defined BASE_PY (
    where python >nul 2>&1
    if %ERRORLEVEL%==0 (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
        if %ERRORLEVEL%==0 set "BASE_PY=python"
    )
)

if not defined BASE_PY (
    echo [ERROR] Python 3.11 or newer was not found on PATH.
    echo Please install Python 3.11+ ^(recommended 3.13^) from https://www.python.org/downloads/
    echo and check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [PMP Agent] Creating virtual environment at "%~dp0.venv"...
    %BASE_PY% -m venv "%~dp0.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [PMP Agent] Reusing existing virtual environment at "%~dp0.venv".
)

echo [PMP Agent] Checking Python version inside .venv...
"%~dp0.venv\Scripts\python.exe" --version

echo [PMP Agent] Installing runtime dependencies ^(requirements.txt^)...
"%~dp0.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo [ERROR] Failed to install runtime dependencies from requirements.txt.
    pause
    exit /b 1
)

if "%INSTALL_DEV%"=="1" (
    echo [PMP Agent] Installing dev/browser verification dependencies ^(requirements-dev.txt^)...
    "%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements-dev.txt"
    if errorlevel 1 (
        echo [ERROR] Failed to install dev dependencies from requirements-dev.txt.
        pause
        exit /b 1
    )
    echo [PMP Agent] Ensuring Playwright Chromium browser is installed...
    "%~dp0.venv\Scripts\python.exe" -m playwright install chromium
    if errorlevel 1 (
        echo [ERROR] Failed to install Playwright Chromium.
        pause
        exit /b 1
    )
)

echo.
echo [PMP Agent] Setup complete!
echo  - Launch app:               start_windows.bat
echo  - Launch offline demo mode: start_offline.bat
echo  - Run verification:         verify_windows.bat
if "%INSTALL_DEV%"=="0" (
    echo.
    echo Optional full verification ^(ruff, mypy, Playwright browser UI tests^):
    echo   setup_windows.bat --dev
)
echo.
echo Note: Any edits you make to .env require restarting the app to take effect.
endlocal
