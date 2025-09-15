@echo off
REM Setup script for Audio Deepfake Detection using FMSL (Windows)

echo 🎓 Setting up Audio Deepfake Detection using FMSL...

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Install package in development mode
echo 🔧 Installing package in development mode...
pip install -e .

REM Create necessary directories
echo 📁 Creating directories...
if not exist "data\ASVspoof2019\LA" mkdir "data\ASVspoof2019\LA"
if not exist "models" mkdir "models"
if not exist "results" mkdir "results"
if not exist "logs" mkdir "logs"

REM Download example data (optional)
echo 📥 Creating example data...
if not exist "data\example_audio.wav" (
    python -c "import numpy as np; import soundfile as sf; import os; os.makedirs('data', exist_ok=True); audio = np.zeros(16000); sf.write('data/example_audio.wav', audio, 16000); print('Created example audio file')"
)

echo ✅ Setup completed successfully!
echo.
echo 🚀 Next steps:
echo 1. Activate virtual environment: venv\Scripts\activate.bat
echo 2. Download ASVspoof2019 LA dataset to data\ASVspoof2019\LA\
echo 3. Run training: python Thesis\01_Models\01_Baseline_Models\maze1.py --help
echo 4. Open Jupyter notebook: jupyter notebook Thesis\08_Notebooks\Complete_Thesis.ipynb
echo.
echo 📚 Documentation:
echo - Installation: Thesis\05_Documentation\INSTALLATION.md
echo - Usage: Thesis\05_Documentation\USAGE.md
echo - Troubleshooting: Thesis\05_Documentation\TROUBLESHOOTING.md

pause
