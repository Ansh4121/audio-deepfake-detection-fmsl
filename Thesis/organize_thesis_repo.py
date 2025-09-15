#!/usr/bin/env python3
"""
Organize thesis repository with proper folder structure and file organization
"""

import os
import shutil
from pathlib import Path

def create_folder_structure():
    """Create organized folder structure"""
    
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
    """Organize files into appropriate folders"""
    
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

if __name__ == "__main__":
    print("Creating thesis repository structure...")
    create_folder_structure()
    print("\nOrganizing files...")
    organize_files()
    print("\nThesis repository organization completed!")