@echo off
setlocal
cd /d "%~dp0"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [ERROR] Local virtual environment not found at "%~dp0.venv".
    echo Run setup_windows.bat first.
    pause
    exit /b 1
)

set "PY=%~dp0.venv\Scripts\python.exe"
set "PMP_RUN_LIVE_TESTS="
set "PARTIAL_VERIFICATION=0"

echo [1/4] Secret scan...
"%PY%" "%~dp0scripts\check_secrets.py"
if errorlevel 1 exit /b 1

echo [2/4] Ruff lint...
"%PY%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('ruff') else 1)"
if %ERRORLEVEL%==0 (
    "%PY%" -m ruff check "%~dp0."
    if errorlevel 1 exit /b 1
) else (
    echo   [SKIP] ruff not installed ^(run setup_windows.bat --dev to include lint checks^).
    set "PARTIAL_VERIFICATION=1"
)

echo [3/4] Mypy type check...
"%PY%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('mypy') else 1)"
if %ERRORLEVEL%==0 (
    "%PY%" -m mypy --config-file "%~dp0pyproject.toml"
    if errorlevel 1 exit /b 1
) else (
    echo   [SKIP] mypy not installed ^(run setup_windows.bat --dev to include type checks^).
    set "PARTIAL_VERIFICATION=1"
)

echo [4/4] Offline test suite...
"%PY%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest') else 1)"
if errorlevel 1 (
    echo [INFO] pytest is not installed in .venv. Run: setup_windows.bat --dev
    exit /b 1
)

"%PY%" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('playwright') else 1)"
if %ERRORLEVEL%==0 (
    "%PY%" -m pytest -q
    if errorlevel 1 exit /b 1
) else (
    echo   [WARN] playwright not installed - omitting 3 offline browser UI tests ^(tests\test_ui_browser.py^).
    echo          Run "setup_windows.bat --dev" to install dev/browser deps for full 122-test verification.
    set "PARTIAL_VERIFICATION=1"
    "%PY%" -m pytest -q --ignore="%~dp0tests\test_ui_browser.py"
    if errorlevel 1 exit /b 1
)

echo.
if "%PARTIAL_VERIFICATION%"=="1" (
    echo [PMP Agent] Partial runtime verification succeeded ^(some dev/browser checks skipped; run setup_windows.bat --dev for full coverage^).
) else (
    echo [PMP Agent] Full verification succeeded ^(secret scan, ruff, mypy, and all offline/browser tests^)!
)
endlocal
