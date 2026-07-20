@echo off
rem One-click retrieval precision/recall check (Windows).
rem Runs the pytest gate, then the human-readable per-query report.
cd /d "%~dp0..\.."

if not exist venv\Scripts\python.exe (
    echo No venv found at venv\. Run run.bat first to set up the project.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

echo ============================================
echo  Pytest gate (pass/fail)
echo ============================================
python -m pytest assistant\tests\test_retrieval_eval.py -v
echo.
echo ============================================
echo  Per-query report (lexical)
echo ============================================
python -m assistant.tests.eval_retrieval lexical
echo.
echo ============================================
echo  Per-query report (vector, if installed)
echo ============================================
python -m assistant.tests.eval_retrieval vector

pause
