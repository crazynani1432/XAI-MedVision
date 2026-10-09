@echo off
TITLE XAI-MedVision Server Launcher
echo =========================================================================
echo    XAI-MedVision: Multi-Specialty Explainable AI Diagnostic Platform
echo =========================================================================
echo.

IF EXIST venv\Scripts\activate.bat (
    echo [*] Activating Python 3.11 virtual environment...
    call venv\Scripts\activate.bat
) ELSE (
    echo [!] Virtual environment not found. Using system Python...
)

echo [*] Pre-warming models and launching FastAPI application...
echo [*] Web Interface: http://localhost:8000/
echo [*] API Documentation: http://localhost:8000/docs
echo.
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
