@echo off
echo Copying essential files for GitHub repository...

REM Copy remaining maze models
copy ..\maze3_fmsl_standardized.py models\
copy ..\maze4.py models\
copy ..\maze4_fmsl_standardized.py models\
copy ..\maze5.py models\
copy ..\maze5_fmsl_standardized.py models\
copy ..\maze6.py models\
copy ..\maze6_fmsl_standardized.py models\
copy ..\maze7.py models\
copy ..\maze7_fmsl_standardized.py models\
copy ..\maze8.py models\
copy ..\maze8_fmsl_standardized.py models\

REM Copy FMSL core files
copy ..\fmsl_advanced.py utils\
copy ..\fmsl_standardized_config.py utils\

REM Copy evaluation files
copy ..\Maze1_eval.py evaluation\
copy ..\Maze2_Eval.py evaluation\
copy ..\Maze3_eval.py evaluation\
copy ..\Maze5_eval.py evaluation\
copy ..\Maze6_Eval.py evaluation\
copy ..\Maze7_eval.py evaluation\
copy ..\Maze8_eval.py evaluation\

REM Copy config files
copy ..\model_config.yaml configs\
copy ..\model_config_Maze6.yaml configs\
copy ..\model_config_Model6_FMSL.yaml configs\
copy ..\model_config_RawNet.yaml configs\

REM Copy documentation
copy ..\GOOGLE_COLAB_GUIDE.md docs\
copy ..\MAZE5_EVALUATION_README.md docs\
copy ..\MULTI_MODEL_COMPARISON_README.md docs\

REM Copy main evaluation script
copy ..\Main_eval.py .

echo All essential files copied successfully!
pause
