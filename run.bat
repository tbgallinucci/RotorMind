@echo off
setlocal enabledelayedexpansion
rem Rotordynamics Copilot - one-click launcher (Windows)
rem Needs Python 3.10+. Finds one via the py launcher, creates/repairs the
rem venv, installs the package, opens the browser.
cd /d "%~dp0"

rem ---- find a Python >= 3.10 ----
set "PYCMD="
for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PYCMD (
        py -%%V -c "exit()" >nul 2>&1 && set "PYCMD=py -%%V"
    )
)
if not defined PYCMD (
    python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
    echo No Python 3.10+ found. Your default 'python' may be older ^(e.g. 3.9^).
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo ^(keep "py launcher" checked in the installer^), then run this again.
    pause
    exit /b 1
)
echo [setup] using: %PYCMD%

rem ---- venv: recreate if missing or built with an old Python ----
set "NEED_VENV=1"
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set "NEED_VENV="
)
if defined NEED_VENV (
    if exist venv (
        echo [setup] existing venv uses an unsupported Python - rebuilding it...
        rmdir /s /q venv
    )
    echo [setup] creating virtual environment...
    %PYCMD% -m venv venv || goto :error
    call venv\Scripts\activate.bat
    echo [setup] installing package ^(first run takes a few minutes^)...
    python -m pip install --upgrade pip >nul
    pip install -e . || goto :error
) else (
    call venv\Scripts\activate.bat
    pip show rotordynamics-copilot >nul 2>&1 || pip install -e . || goto :error
)

echo [run] starting Rotordynamics Copilot at http://localhost:8000
echo [run] chat needs LM Studio serving on http://localhost:1234 (wiki browsing works without it)
start "" http://localhost:8000
python -m uvicorn assistant.app.main:app --host 127.0.0.1 --port 8000
goto :eof

:error
echo Setup failed. Scroll up for the first error message.
pause
