@echo off
title AI-Based Diabetes Prediction System
echo ========================================================
echo   AI-BASED DIABETES PREDICTION SYSTEM (FINAL YEAR PROJECT)
echo   Student: Pranali
echo ========================================================
echo.
echo Starting Streamlit Application...
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
) else (
    python -m streamlit run app.py --server.port 8501 --server.headless false
)

pause
