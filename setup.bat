@echo off
echo ============================================
echo   SAHAYAK - Setup Script
echo   Team Caraxes - SIH 2026
echo ============================================
echo.

REM Create virtual environment
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists.
)

REM Activate venv
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo [3/4] Installing dependencies...
pip install -r requirements.txt

REM Generate data and train model
echo [4/4] Generating synthetic data and training model...
python data\generate_synthetic_data.py
python model\train_model.py

echo.
echo ============================================
echo   Setup complete!
echo   Run: streamlit run app.py
echo ============================================
pause
