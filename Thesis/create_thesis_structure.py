#!/usr/bin/env python3
"""
Create organized thesis repository structure
"""

import os
import shutil
from pathlib import Path

def create_folder_structure():
    """Create the complete thesis folder structure"""
    
    # Create main directories
    directories = [
        "Thesis/01_Models/01_Baseline_Models",
        "Thesis/01_Models/02_FMSL_Enhanced_Models", 
        "Thesis/02_Evaluation_Scripts",
        "Thesis/03_Data_Processing",
        "Thesis/04_Results_and_Analysis",
        "Thesis/05_Documentation",
        "Thesis/06_Utilities",
        "Thesis/07_Configuration_Files",
        "Thesis/08_Notebooks",
        "Thesis/09_LaTeX_Source"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

def organize_files():
    """Organize files into the thesis structure"""
    
    # File organization mapping
    file_mapping = {
        # Main files
        "README.md": "Thesis/",
        "standardized_maze_config.py": "Thesis/",
        "check_maze_consistency.py": "Thesis/",
        "organize_thesis_repo.py": "Thesis/",
        "verify_maze_configurations.py": "Thesis/",
        "analyze_maze_configurations.py": "Thesis/",
        "create_thesis_structure.py": "Thesis/",
        
        # Baseline Models
        "maze1.py": "Thesis/01_Models/01_Baseline_Models/",
        "maze2.py": "Thesis/01_Models/01_Baseline_Models/",
        "maze3.py": "Thesis/01_Models/01_Baseline_Models/",
        "maze4.py": "Thesis/01_Models/01_Baseline_Models/",
        "maze5.py": "Thesis/01_Models/01_Baseline_Models/",
        "maze6.py": "Thesis/01_Models/01_Baseline_Models/",
        "maze7.py": "Thesis/01_Models/01_Baseline_Models/",
        "maze8.py": "Thesis/01_Models/01_Baseline_Models/",
        "main.py": "Thesis/01_Models/01_Baseline_Models/",
        
        # FMSL Enhanced Models
        "maze1_fmsl.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze1_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze2_fmsl.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze2_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze3_fmsl.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze3_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze4_fmsl.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze4_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze5_fmsl.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze5_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze6_fmsl.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze6_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze7_fmsl.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze7_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze8_fmsl.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "maze8_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        "main_fmsl_standardized.py": "Thesis/01_Models/02_FMSL_Enhanced_Models/",
        
        # Evaluation Scripts
        "Maze1_eval.py": "Thesis/02_Evaluation_Scripts/",
        "Maze2_Eval.py": "Thesis/02_Evaluation_Scripts/",
        "Maze3_eval.py": "Thesis/02_Evaluation_Scripts/",
        "Maze5_eval.py": "Thesis/02_Evaluation_Scripts/",
        "Maze5_eval_clean.py": "Thesis/02_Evaluation_Scripts/",
        "Maze6_Eval.py": "Thesis/02_Evaluation_Scripts/",
        "Maze7_eval.py": "Thesis/02_Evaluation_Scripts/",
        "Maze8_eval.py": "Thesis/02_Evaluation_Scripts/",
        "Main_eval.py": "Thesis/02_Evaluation_Scripts/",
        
        # Data Processing
        "colab_setup.py": "Thesis/03_Data_Processing/",
        "colab_quick_setup.py": "Thesis/03_Data_Processing/",
        "setup_asvspoof2021.py": "Thesis/03_Data_Processing/",
        "setup_asvspoof2021_paths.py": "Thesis/03_Data_Processing/",
        "download_la_keys.py": "Thesis/03_Data_Processing/",
        
        # Results and Analysis
        "evaluation_results/": "Thesis/04_Results_and_Analysis/",
        "thesis_results/": "Thesis/04_Results_and_Analysis/",
        "thesis_final_results/": "Thesis/04_Results_and_Analysis/",
        "thesis_analysis_results/": "Thesis/04_Results_and_Analysis/",
        "comprehensive_fmsl_analysis.py": "Thesis/04_Results_and_Analysis/",
        "thesis_automation.py": "Thesis/04_Results_and_Analysis/",
        "maze5_comparison_visualizer.py": "Thesis/04_Results_and_Analysis/",
        
        # Documentation
        "GOOGLE_COLAB_GUIDE.md": "Thesis/05_Documentation/",
        "MAZE5_EVALUATION_GUIDE.md": "Thesis/05_Documentation/",
        "THESIS_ANALYSIS_USAGE_GUIDE.md": "Thesis/05_Documentation/",
        "ENHANCED_VISUALIZATIONS_SUMMARY.md": "Thesis/05_Documentation/",
        "FOCUSED_VISUALIZATIONS_SUMMARY.md": "Thesis/05_Documentation/",
        "FIGURE2_BOTTLENECK_ANALYSIS.md": "Thesis/05_Documentation/",
        "FIGURE3_FMSL_SOLUTION.md": "Thesis/05_Documentation/",
        "THESIS_INTEGRATION_SUMMARY.md": "Thesis/05_Documentation/",
        "MULTI_MODEL_COMPARISON_README.md": "Thesis/05_Documentation/",
        "MAZE5_COMPARISON_README.md": "Thesis/05_Documentation/",
        "MAZE5_EVALUATION_README.md": "Thesis/05_Documentation/",
        "MAZE5_EVALUATION_ISSUES_AND_SOLUTIONS.md": "Thesis/05_Documentation/",
        
        # Utilities
        "fmsl_advanced.py": "Thesis/06_Utilities/",
        "fmsl_standardized_config.py": "Thesis/06_Utilities/",
        "checkpoint_manager.py": "Thesis/06_Utilities/",
        "unified_model_evaluator.py": "Thesis/06_Utilities/",
        "flexible_model_comparison.py": "Thesis/06_Utilities/",
        "score_file_processor.py": "Thesis/06_Utilities/",
        
        # Configuration Files
        "model_config.yaml": "Thesis/07_Configuration_Files/",
        "model_config_Maze6.yaml": "Thesis/07_Configuration_Files/",
        "model_config_Model6_FMSL.yaml": "Thesis/07_Configuration_Files/",
        "model_config_RawNet.yaml": "Thesis/07_Configuration_Files/",
        "models_config_template.json": "Thesis/07_Configuration_Files/",
        
        # Notebooks
        "Colab_CPU_Test_Notebook.ipynb": "Thesis/08_Notebooks/",
        "FMSL_Maze_Models_Colab.ipynb": "Thesis/08_Notebooks/",
        "GOOGLE_COLAB_SEQUENTIAL_TRAINING.ipynb": "Thesis/08_Notebooks/",
        
        # LaTeX Source
        "WUT-Thesis/": "Thesis/09_LaTeX_Source/",
        "Thesis.pdf": "Thesis/09_LaTeX_Source/",
        "Audio Deepfake Detection FMSL Research.pdf": "Thesis/09_LaTeX_Source/",
        "Thesis Improvement and Citation Search.pdf": "Thesis/09_LaTeX_Source/",
        "Criteria for the correctness of synchronization solutions (1).pdf": "Thesis/09_LaTeX_Source/",
        "Graduate info card.pdf": "Thesis/09_LaTeX_Source/",
    }
    
    # Copy files to organized structure
    for source, destination in file_mapping.items():
        if os.path.exists(source):
            if os.path.isdir(source):
                # Copy directory
                dest_path = os.path.join(destination, source)
                if not os.path.exists(dest_path):
                    shutil.copytree(source, dest_path)
                    print(f"Copied directory: {source} -> {dest_path}")
            else:
                # Copy file
                dest_path = os.path.join(destination, os.path.basename(source))
                if not os.path.exists(dest_path):
                    shutil.copy2(source, dest_path)
                    print(f"Copied file: {source} -> {dest_path}")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt
*.pkl
*.pickle

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/
tensorboard_logs/

# Data
data/
datasets/
*.wav
*.flac
*.mp3

# Models
models/
checkpoints/
saved_models/

# Results
results/
outputs/
evaluation_results/
thesis_results/

# Temporary files
*.tmp
*.temp
temp/
tmp/

# LaTeX
*.aux
*.bbl
*.blg
*.fdb_latexmk
*.fls
*.log
*.out
*.synctex.gz
*.toc
*.lof
*.lot
"""
    
    with open('Thesis/.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("Created .gitignore file")

def create_license():
    """Create LICENSE file"""
    license_content = """MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open('Thesis/LICENSE', 'w') as f:
        f.write(license_content)
    print("Created LICENSE file")

if __name__ == "__main__":
    print("Creating thesis repository structure...")
    create_folder_structure()
    print("\nOrganizing files...")
    organize_files()
    print("\nCreating additional files...")
    create_gitignore()
    create_license()
    print("\nThesis repository organization completed!")
    print("\nRepository structure created at: Thesis/")