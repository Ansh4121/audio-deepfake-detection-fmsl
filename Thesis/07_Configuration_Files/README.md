# Configuration Files Documentation

This directory contains configuration files for model training and the standardized parameters used for thesis results.

## Overview

Two types of configuration are used in this project:

1. **Python-based standardization** (`standardized_maze_config.py`) - **Used for thesis results**
2. **YAML configuration files** (this directory) - For optional separate training setups

## Files in This Directory

### YAML Configuration Files

| File | Description | Used for Thesis Results? |
|------|-------------|-------------------------|
| `model_config_Maze5.yaml` | Maze 5 training configuration | No |
| `model_config_Maze6.yaml` | Maze 6 training configuration | No |
| `model_config_Model4.yaml` | Model 4 training configuration | No |
| `model_config_Model7.yaml` | Model 7 training configuration | No |
| `model_config_RawNet.yaml` | RawNet baseline configuration | No |
| `requirements.txt` | Python dependencies | Yes |

### Python Standardization (Used for Thesis)

The actual standardization used for all thesis results is in:
- `../standardized_maze_config.py` - Main configuration
- `../06_Utilities/fmsl_advanced.py` - FMSL system implementation
- `../06_Utilities/fmsl_standardized_config.py` - FMSL configuration

## Why YAML Files Were Not Used

The YAML files in this directory were **not** used for generating thesis results because:

1. **Machine-specific paths**: YAMLs contain paths like `/content/drive/MyDrive/...` that are specific to Google Colab environments
2. **Per-model variations**: Each YAML has different hyperparameters that would make comparison unfair
3. **Consistency requirement**: The thesis documents a single standardized protocol (Chapter 6)

## Standardized Configuration (Python)

All thesis results use these standardized parameters:

```python
# Architecture (same for baseline and FMSL)
filts = [128, [128, 128], [128, 256]]
nb_fc_node = 1024
nb_classes = 2
sample_rate = 16000
first_conv = 251
dropout_rate = 0.3

# Wav2Vec2 (for models using it)
wav2vec2_model_name = 'facebook/wav2vec2-base-960h'
wav2vec2_output_dim = 768
wav2vec2_freeze = True

# FMSL Parameters
fmsl_type = 'prototype'
fmsl_n_prototypes = 3
fmsl_s = 32.0  # Scale factor (AM-Softmax)
fmsl_m = 0.45  # Angular margin

# Training
batch_size = 12
lr = 0.0001
weight_decay = 0.0001
num_epochs = 5
seed = 1234
```

## YAML File Format

Each YAML file follows this structure:

```yaml
model:
  # Model architecture parameters
  wav2vec2_model_name: 'facebook/wav2vec2-base-960h'
  projected_dim: 768
  filts: [128, [128, 128], [128, 256]]
  nb_fc_node: 1024
  nb_classes: 2
  # ... more parameters

training:
  num_epochs: 100
  batch_size: 16
  learning_rate: 0.0001
  # ... more parameters
```

## Using YAML Files (Optional)

If you want to use YAML files for custom training:

```python
import yaml

with open('model_config_Maze5.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Use config['model'] and config['training']
```

**Note**: Update the paths in YAML files to match your environment before use.

## Reproducibility

For reproducing thesis results:

1. Use `../standardized_maze_config.py` for configuration
2. Use `../06_Utilities/fmsl_advanced.py` for FMSL implementation
3. Run evaluation scripts from `../02_Evaluation_Scripts/`

The YAML files are provided for reference and for users who want to experiment with different hyperparameters.
