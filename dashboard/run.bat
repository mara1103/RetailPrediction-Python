@echo off
REM ========================================
REM Streamlit Dashboard - Install & Run
REM Optimizare Stocuri - LSTM Forecast
REM ========================================

echo.
echo ==========================================
echo   DASHBOARD OPTIMIZARE STOCURI - SETUP
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python nu este instalat sau nu este in PATH
    echo Va rog instaleaza Python de la https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python detectat

REM Check if venv exists, create if not
if not exist "venv" (
    echo.
    echo Creare virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Nu am putut crea virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment creat
)

REM Activate venv
echo.
echo Activare virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Nu am putut activa virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activ

REM Install requirements
echo.
echo Instalare dependinte (prima data va lua 2-5 minute)...
echo Asteptati...
pip install -q --upgrade pip
pip install -q setuptools wheel
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Nu am putut instala dependinte
    pause
    exit /b 1
)
echo [OK] Dependinte instalate

REM Run Streamlit app
echo.
echo ==========================================
echo   PORNIRE APLICATIE
echo ==========================================
echo.
echo Dashboard va fi disponibil la:
echo   http://localhost:8501
echo.
echo Apasa CTRL+C pentru a opri aplicatia
echo.
pause

streamlit run app.py
