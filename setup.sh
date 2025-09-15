#!/bin/bash
# Setup script for Audio Deepfake Detection using FMSL

set -e

echo "🎓 Setting up Audio Deepfake Detection using FMSL..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required, found $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/ASVspoof2019/LA
mkdir -p models
mkdir -p results
mkdir -p logs

# Set up Git hooks (optional)
if [ -d ".git" ]; then
    echo "🔗 Setting up Git hooks..."
    cp scripts/pre-commit .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit
fi

# Download example data (optional)
echo "📥 Downloading example data..."
if [ ! -f "data/example_audio.wav" ]; then
    # Create a simple example audio file
    python -c "
import numpy as np
import soundfile as sf
import os
os.makedirs('data', exist_ok=True)
# Generate 1 second of silence
audio = np.zeros(16000)
sf.write('data/example_audio.wav', audio, 16000)
print('Created example audio file')
"
fi

echo "✅ Setup completed successfully!"
echo ""
echo "🚀 Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download ASVspoof2019 LA dataset to data/ASVspoof2019/LA/"
echo "3. Run training: python Thesis/01_Models/01_Baseline_Models/maze1.py --help"
echo "4. Open Jupyter notebook: jupyter notebook Thesis/08_Notebooks/Complete_Thesis.ipynb"
echo ""
echo "📚 Documentation:"
echo "- Installation: Thesis/05_Documentation/INSTALLATION.md"
echo "- Usage: Thesis/05_Documentation/USAGE.md"
echo "- Troubleshooting: Thesis/05_Documentation/TROUBLESHOOTING.md"
