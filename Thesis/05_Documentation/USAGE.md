# Usage Guide

## Quick Start

### 1. Training Models

#### Baseline Models
```bash
# Train Maze1 (Baseline RawNet2)
python Thesis/01_Models/01_Baseline_Models/maze1.py \
    --database_path data/ASVspoof2019/LA/ \
    --protocols_path data/ASVspoof2019/LA/protocols/ \
    --track LA \
    --batch_size 32 \
    --num_epochs 5

# Train Maze2 (Wav2Vec2 + SpecAugment)
python Thesis/01_Models/01_Baseline_Models/maze2.py \
    --database_path data/ASVspoof2019/LA/ \
    --protocols_path data/ASVspoof2019/LA/protocols/ \
    --track LA \
    --batch_size 128 \
    --num_epochs 5
```

#### FMSL-Enhanced Models
```bash
# Train Maze1 with FMSL
python Thesis/01_Models/02_FMSL_Enhanced_Models/maze1_fmsl_standardized.py \
    --database_path data/ASVspoof2019/LA/ \
    --protocols_path data/ASVspoof2019/LA/protocols/ \
    --track LA \
    --batch_size 32 \
    --num_epochs 5

# Train Maze2 with FMSL
python Thesis/01_Models/02_FMSL_Enhanced_Models/maze2_fmsl_standardized.py \
    --database_path data/ASVspoof2019/LA/ \
    --protocols_path data/ASVspoof2019/LA/protocols/ \
    --track LA \
    --batch_size 128 \
    --num_epochs 5
```

### 2. Evaluating Models

```bash
# Evaluate single model
python Thesis/02_Evaluation_Scripts/Maze1_eval.py \
    --model_type maze1 \
    --model_path models/maze1_best.pth \
    --batch_size 128

# Comprehensive evaluation
python Thesis/02_Evaluation_Scripts/comprehensive_evaluation.py \
    --data_dir data/ASVspoof2019/LA/ \
    --protocol_file data/ASVspoof2019/LA/protocols/ASVspoof2019.LA.cm.eval.trl.txt \
    --output_dir results/ \
    --batch_size 128
```

### 3. Running Analysis

```bash
# Generate thesis analysis
python Thesis/02_Evaluation_Scripts/comprehensive_thesis_analyser.py

# Process score files
python Thesis/02_Evaluation_Scripts/score_file_processor.py \
    --data_dir results/ \
    --protocol_file data/ASVspoof2019/LA/protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

## Configuration

### Model Configuration
Edit configuration files in `Thesis/07_Configuration_Files/`:
- `model_config_RawNet.yaml` - RawNet configuration
- `model_config_Maze5.yaml` - Maze5 configuration
- `standardized_maze_config.py` - Standardized configurations

### Training Parameters
- `batch_size`: Batch size for training (default: 32)
- `num_epochs`: Number of training epochs (default: 5)
- `lr`: Learning rate (default: 0.0001)
- `weight_decay`: Weight decay (default: 0.0001)

### FMSL Parameters
- `fmsl_type`: FMSL type ('prototype')
- `fmsl_n_prototypes`: Number of prototypes (default: 3)
- `fmsl_s`: Scale parameter (default: 32.0)
- `fmsl_m`: Margin parameter (default: 0.45)

## Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open complete thesis notebook
# Navigate to Thesis/08_Notebooks/Complete_Thesis.ipynb
```

## Command Line Interface

```bash
# Train with CLI
fmsl-train --model maze1 --data_path data/ --epochs 5

# Evaluate with CLI
fmsl-eval --model_path models/maze1.pth --data_path data/
```

## Examples

See `examples/` directory for:
- Basic usage examples
- Advanced configuration examples
- Custom model implementations
- Evaluation scripts
