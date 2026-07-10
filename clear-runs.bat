@echo off
rem Remove ALL simulation run reports from the wiki (theory pages untouched).
cd /d "%~dp0"
echo This deletes every run page, its plots, and its index entries.
set /p OK="Continue? [y/N]: "
if /i not "%OK%"=="y" (echo Cancelled. & pause & exit /b 0)
if exist venv\Scripts\python.exe (venv\Scripts\python.exe scripts\clear_runs.py) else (python scripts\clear_runs.py)
pause
