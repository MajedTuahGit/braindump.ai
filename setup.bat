@echo off
REM BrainDump.AI — Windows one-command setup
REM Run this once: setup.bat

echo.
echo ============================================================
echo   BrainDump.AI — Setup
echo ============================================================
echo.

REM 1. Create virtual environment
echo [1/5] Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

REM 2. Install dependencies
echo [2/5] Installing Python packages...
pip install --upgrade pip -q
pip install -r requirements.txt -q

REM 3. Download spaCy model
echo [3/5] Downloading spaCy language model...
python -m spacy download en_core_web_sm

REM 4. Download NLTK data (used in augment.py)
echo [4/5] Downloading NLTK data...
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

REM 5. Create required directories
echo [5/5] Creating directories...
mkdir models 2>NUL
mkdir db 2>NUL
mkdir frontend 2>NUL

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Start the server:
echo      .venv\Scripts\uvicorn.exe api.main:app --reload --port 8000
echo.
echo   2. Open your browser:
echo      http://localhost:8000
echo.
echo   3. (Optional) Train Tier 1 for better accuracy:
echo      python training/train_tier1.py
echo.
pause
