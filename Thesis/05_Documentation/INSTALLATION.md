# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM recommended
- 20GB+ free disk space

## Quick Installation

### Option 1: Using pip
```bash
# Clone the repository
git clone https://github.com/yourusername/audio-deepfake-detection-fmsl.git
cd audio-deepfake-detection-fmsl

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Using conda
```bash
# Clone the repository
git clone https://github.com/yourusername/audio-deepfake-detection-fmsl.git
cd audio-deepfake-detection-fmsl

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate audio-deepfake-fmsl
```

### Option 3: Using Docker
```bash
# Build Docker image
docker build -t audio-deepfake-fmsl .

# Run container
docker run -it --gpus all audio-deepfake-fmsl
```

## Dataset Setup

1. Download ASVspoof2019 LA dataset
2. Extract to `data/ASVspoof2019/LA/`
3. Update paths in configuration files

## Verification

```bash
# Test installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Run basic test
python -m pytest tests/test_basic.py
```

## Troubleshooting

### Common Issues:
- **CUDA not available**: Install CUDA toolkit and compatible PyTorch version
- **Audio loading errors**: Install librosa and soundfile
- **Memory issues**: Reduce batch size in configuration files
- **Import errors**: Ensure all dependencies are installed

### Getting Help:
- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Open an issue on GitHub
- Contact: your.email@university.edu
