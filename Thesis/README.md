# Audio Deepfake Detection using Frequency-Modulated Spectral Loss (FMSL)

## Overview

This repository contains the complete implementation and evaluation of a novel Frequency-Modulated Spectral Loss (FMSL) approach for audio deepfake detection using the **ASVspoof2019 LA dataset**. The research presents a comprehensive comparison between baseline models and FMSL-enhanced models across 8 different architectural configurations.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/audio-deepfake-detection-fmsl.git
cd audio-deepfake-detection-fmsl

# Setup environment (Linux/Mac)
chmod +x setup.sh
./setup.sh

# Setup environment (Windows)
setup.bat

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate.bat  # Windows

# Run basic tests
python tests/test_basic.py
```

## 📊 Dataset

- **ASVspoof2019 LA** (Logical Access track)
- **Training**: 25,380 bonafide + 25,380 spoofed utterances
- **Evaluation**: 7,355 bonafide + 63,882 spoofed utterances
- **Sample Rate**: 16 kHz
- **Format**: FLAC audio files

## 📁 Repository Structure

```
Thesis_Project/
├── 📦 Essential Files
│   ├── .gitignore                   # Git ignore rules
│   ├── requirements.txt             # Python dependencies
│   ├── setup.py                     # Package installation
│   ├── environment.yml              # Conda environment
│   ├── setup.sh                     # Linux/Mac setup script
│   ├── setup.bat                    # Windows setup script
│   └── tests/                       # Test suite
│       ├── __init__.py
│       └── test_basic.py            # Basic functionality tests
│
├── 🧠 Models
│   ├── 01_Baseline_Models/          # Original models without FMSL
│   │   ├── maze1.py                 # Baseline RawNet2 with Fixed SincConv
│   │   ├── maze2.py                 # Wav2Vec2 + SpecAugment + Focal Loss
│   │   ├── maze3.py                 # RawNetSinc + SE Transformer
│   │   ├── maze4.py                 # RawNetSinc + SpecAugment
│   │   ├── maze5.py                 # RawNetSinc + SpecAugment + Focal Loss
│   │   ├── maze6.py                 # RawNet + Wav2Vec2 + Transformer
│   │   ├── maze7.py                 # RawNet + Wav2Vec2 + SpecAugment + Focal Loss
│   │   ├── maze8.py                 # Advanced RawNet + Wav2Vec2 + Transformer
│   │   └── main.py                  # Main training script
│   └── 02_FMSL_Enhanced_Models/     # FMSL-enhanced versions
│       ├── maze1_fmsl.py            # Maze1 + FMSL
│       ├── maze1_fmsl_standardized.py
│       ├── maze2_fmsl.py            # Maze2 + FMSL
│       ├── maze2_fmsl_standardized.py
│       ├── ...                      # All other FMSL variants
│       └── main_fmsl_standardized.py
├── 02_Evaluation_Scripts/           # Model evaluation scripts
│   ├── Maze1_eval.py
│   ├── Maze2_Eval.py
│   ├── ...                          # All evaluation scripts
│   └── Main_eval.py
├── 03_Data_Processing/              # Data preparation and processing
│   ├── colab_setup.py
│   ├── setup_asvspoof2021.py
│   └── download_la_keys.py
├── 04_Results_and_Analysis/         # Results, analysis, and visualizations
│   ├── evaluation_results/
│   ├── thesis_results/
│   ├── comprehensive_fmsl_analysis.py
│   └── thesis_automation.py
├── 📚 Documentation/                # Complete documentation
│   ├── INSTALLATION.md              # Installation guide
│   ├── USAGE.md                     # Usage guide
│   ├── TROUBLESHOOTING.md           # Troubleshooting guide
│   ├── GOOGLE_COLAB_GUIDE.md
│   ├── MAZE5_EVALUATION_GUIDE.md
│   ├── THESIS_ANALYSIS_USAGE_GUIDE.md
│   └── ...                          # All documentation files
├── 🔧 Utilities/                    # Utility functions and tools
│   ├── fmsl_advanced.py
│   ├── fmsl_standardized_config.py
│   ├── checkpoint_manager.py
│   ├── unified_model_evaluator.py
│   ├── data_preprocessor.py         # Data preprocessing utilities
│   └── model_trainer.py             # Automated training script
├── 07_Configuration_Files/          # Model configuration files
│   ├── model_config.yaml
│   ├── model_config_Maze6.yaml
│   └── models_config_template.json
├── 📓 Notebooks/                    # Jupyter notebooks
│   ├── Complete_Thesis.ipynb        # Main thesis notebook
│   ├── Colab_CPU_Test_Notebook.ipynb
│   ├── FMSL_Maze_Models_Colab.ipynb
│   └── GOOGLE_COLAB_SEQUENTIAL_TRAINING.ipynb
└── 📄 LaTeX/                        # LaTeX source and PDFs (in WUT-Thesis/)
    ├── Thesis.pdf
    └── Audio Deepfake Detection FMSL Research.pdf
```

## Key Features

### 1. FMSL (Frequency-Modulated Spectral Loss)
- **Novel Contribution**: A new loss function specifically designed for audio deepfake detection
- **Frequency-Domain Enhancement**: Improves feature learning in the frequency domain
- **Geometric Learning**: Incorporates geometric constraints for better classification

### 2. Comprehensive Model Comparison
- **8 Baseline Models**: From simple RawNet2 to complex Wav2Vec2 + Transformer architectures
- **8 FMSL-Enhanced Models**: Each baseline model enhanced with FMSL
- **Standardized Configurations**: Ensures fair comparison across all models

### 3. Standardized Architecture
All models follow a consistent configuration:
- **Input**: 16kHz audio samples
- **Architecture**: `filts: [128, [128, 128], [128, 256]]`
- **FC Layer**: 1024 nodes
- **Classes**: 2 (bonafide vs spoof)
- **Training**: 5 epochs, batch size 12, AdamW optimizer

## Model Descriptions

### Baseline Models (Maze1-Maze8)

1. **Maze1**: Baseline RawNet2 with Fixed SincConv
   - Fixed (non-trainable) SincConv filters
   - 6 residual blocks
   - Establishes baseline performance

2. **Maze2**: Wav2Vec2 + SpecAugment + Focal Loss
   - Pre-trained Wav2Vec2 feature extractor
   - SpecAugment data augmentation
   - Focal Loss for class imbalance

3. **Maze3**: RawNetSinc + SE Transformer
   - Trainable SincConv filters
   - Squeeze-Excitation blocks
   - Transformer encoder

4. **Maze4**: RawNetSinc + SpecAugment
   - Trainable SincConv
   - SpecAugment augmentation
   - Standard residual architecture

5. **Maze5**: RawNetSinc + SpecAugment + Focal Loss
   - Combines Maze4 with Focal Loss
   - Better handling of class imbalance

6. **Maze6**: RawNet + Wav2Vec2 + Transformer
   - Multi-modal feature fusion
   - Wav2Vec2 + RawNet features
   - Transformer for sequence modeling

7. **Maze7**: RawNet + Wav2Vec2 + SpecAugment + Focal Loss
   - Most comprehensive baseline
   - All advanced techniques combined

8. **Maze8**: Advanced RawNet + Wav2Vec2 + Transformer
   - Latest architectural improvements
   - Optimized for performance

### FMSL-Enhanced Models
Each baseline model has a corresponding FMSL-enhanced version:
- **FMSL Integration**: Novel frequency-domain processing
- **Geometric Learning**: Enhanced feature representation
- **Consistent Architecture**: Same base structure as baseline

## Configuration Verification

### Standardized Parameters
All models use the following standardized configuration:

```python
STANDARDIZED_CONFIG = {
    'architecture': {
        'filts': [128, [128, 128], [128, 256]],
        'nb_fc_node': 1024,
        'nb_classes': 2,
        'sample_rate': 16000,
        'first_conv': 251,
        'dropout_rate': 0.3
    },
    'wav2vec2': {
        'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
        'wav2vec2_output_dim': 768,
        'wav2vec2_freeze': True
    },
    'fmsl': {
        'fmsl_type': 'prototype',
        'fmsl_n_prototypes': 3,
        'fmsl_s': 32.0,
        'fmsl_m': 0.45,
        'fmsl_enable_lsa': False
    },
    'training': {
        'batch_size': 12,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'grad_clip_norm': 1.0,
        'num_epochs': 5,
        'seed': 1234
    }
}
```

### Verification Results
✅ **All maze configurations are consistent with standardized settings**
- Baseline and FMSL models have identical base architectures
- Only FMSL-specific parameters differ between baseline and FMSL versions
- Ensures fair comparison across all model variants

## Usage

### Training Models

```bash
# Train baseline model
python maze1.py --database_path /path/to/data --protocols_path /path/to/protocols

# Train FMSL-enhanced model
python maze1_fmsl.py --database_path /path/to/data --protocols_path /path/to/protocols
```

### Evaluation

```bash
# Evaluate model
python Maze1_eval.py --model_path /path/to/model.pth --eval_output scores.txt
```

### Configuration

All models use standardized configurations defined in `fmsl_standardized_config.py`:

```python
from fmsl_standardized_config import get_standardized_config

# Get baseline configuration
config = get_standardized_config("baseline")

# Get FMSL configuration
config = get_standardized_config("fmsl")
```

## Results

The repository includes comprehensive evaluation results showing:
- **Performance Comparison**: Baseline vs FMSL-enhanced models
- **Statistical Analysis**: Significance testing and confidence intervals
- **Visualization**: ROC curves, confusion matrices, and performance plots
- **Ablation Studies**: Component-wise analysis of FMSL contributions

## Key Findings

1. **FMSL Effectiveness**: FMSL consistently improves performance across all model architectures
2. **Architectural Consistency**: Standardized configurations ensure fair comparison
3. **Frequency-Domain Benefits**: FMSL's frequency-domain processing provides significant advantages
4. **Scalability**: FMSL enhancement works across different model complexities

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- Librosa
- Scikit-learn
- TensorBoard
- Matplotlib
- Seaborn

## Citation

If you use this work, please cite:

```bibtex
@thesis{your_thesis_2024,
  title={Audio Deepfake Detection using Frequency-Modulated Spectral Loss},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please contact [your.email@university.edu].

---

**Note**: This repository contains the complete implementation and evaluation of the FMSL approach for audio deepfake detection, providing a comprehensive framework for reproducible research in this domain.
