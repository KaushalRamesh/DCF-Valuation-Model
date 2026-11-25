@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   DCF Analyzer - Starting Dashboard
echo ============================================

if exist ".venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call ".venv\Scripts\activate.bat"
)

python -c "import streamlit, yfinance, pandas, numpy, plotly, dotenv" >nul 2>&1
if errorlevel 1 (
    echo Installing missing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install dependencies. Please check your Python environment.
        pause
        exit /b 1
    )
)

echo Launching Streamlit on http://localhost:8501 ...
python -m streamlit run app.py
if errorlevel 1 (
    echo Application exited with an error.
    pause
    exit /b 1
)
endlocal
