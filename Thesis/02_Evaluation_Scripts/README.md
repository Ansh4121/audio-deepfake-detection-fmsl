# Evaluation Scripts - Documentation

This directory contains the Python scripts used to generate all results, figures, and tables presented in the thesis.

## Scripts Used for Thesis Results

### Primary Analysis Scripts

- **`comprehensive_thesis_analyser.py`** - Main analysis script that generates:
  - Performance comparison tables (baseline vs FMSL)
  - All thesis figures: `maze_models_comparison.png`, `fmsl_standardization_analysis.png`, `bottleneck_analysis.png`, `trend_visualizations.png`, `comprehensive_histogram.png`
  - Results tables in CSV and LaTeX format (`thesis_results_table.csv`, `thesis_results_table.tex`)
  - Summary statistics JSON files

- **`comprehensive_evaluation.py`** - Full evaluation pipeline for model loading and metric calculation

- **`Eval.py`** - Maze 5-focused evaluation generating:
  - ROC curves, Precision-Recall curves
  - Score distributions
  - Comprehensive analysis dashboard (`comprehensive_analysis_dashboard.png`)
  - Confusion matrix analysis (`confusion_matrix_analysis.png`)

### Per-Model Evaluation Scripts

- **`Maze2_Eval.py`**, **`Maze3_Eval.py`**, **`Maze5_eval.py`**, **`Maze6_Eval.py`**, **`Maze7_Eval.py`**, **`Maze8_Eval.py`**
  - Run inference on ASVspoof 2019 LA evaluation set
  - Generate CM score files for each model (baseline and FMSL variants)
  - Follow standardized protocol: 16 kHz sampling, preprocessing, feature extraction, inference
  - Model renumbering (Maze 4 removed, 5-8 → 5-7) is handled in analysis scripts

### Utility Scripts

- **`score_file_processor.py`** - Processes CM score files and calculates metrics

## Usage

To regenerate thesis figures:

```bash
cd Thesis/02_Evaluation_Scripts
python comprehensive_thesis_analyser.py
```

Outputs will be in `thesis_analysis_results/` directory. Copy PNG files to `WUT-Thesis/img/` for inclusion in the thesis.

## Dependencies

See `../07_Configuration_Files/requirements.txt` or repository root `requirements.txt` for Python dependencies.

## Standardization

All scripts use the Python-based standardization defined in `../standardized_maze_config.py` (not the YAML files in `../07_Configuration_Files/`). This ensures fair comparison between baseline and FMSL-enhanced models.

## Repository Location

All scripts are available at: https://github.com/Ansh4121/audio-deepfake-detection-fmsl/tree/main/Thesis/02_Evaluation_Scripts
