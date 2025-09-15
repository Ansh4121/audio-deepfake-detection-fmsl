#!/usr/bin/env python3
"""
Analyze maze configurations to ensure architectural consistency
"""

import os
import re
import ast
from collections import defaultdict

def extract_model_config(file_path):
    """Extract model configuration from a Python file"""
    config = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Look for model_config patterns
        config_patterns = [
            r'model_config\s*=\s*\{([^}]+)\}',
            r"'filts':\s*\[([^\]]+)\]",
            r"'nb_fc_node':\s*(\d+)",
            r"'nb_classes':\s*(\d+)",
            r"'sample_rate':\s*(\d+)",
            r"'first_conv':\s*(\d+)",
            r"'wav2vec2_output_dim':\s*(\d+)",
            r"'wav2vec2_model_name':\s*['\"]([^'\"]+)['\"]"
        ]
        
        # Extract filts configuration
        filts_match = re.search(r"'filts':\s*\[([^\]]+)\]", content)
        if filts_match:
            filts_str = filts_match.group(1)
            # Parse the filts array
            try:
                config['filts'] = ast.literal_eval(f'[{filts_str}]')
            except:
                config['filts'] = filts_str
        
        # Extract other parameters
        for pattern in config_patterns[1:]:
            match = re.search(pattern, content)
            if match:
                key = pattern.split("'")[1]
                value = match.group(1)
                try:
                    config[key] = int(value)
                except ValueError:
                    try:
                        config[key] = float(value)
                    except ValueError:
                        config[key] = value
        
        return config
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def analyze_all_mazes():
    """Analyze all maze configurations"""
    
    # Find all maze files
    maze_files = []
    for file in os.listdir('.'):
        if file.startswith('maze') and file.endswith('.py') and 'eval' not in file:
            maze_files.append(file)
    
    print("Found maze files:")
    for file in sorted(maze_files):
        print(f"  - {file}")
    
    print("\n" + "="*80)
    print("MAZE CONFIGURATION ANALYSIS")
    print("="*80)
    
    configs = {}
    
    for file in sorted(maze_files):
        print(f"\nAnalyzing: {file}")
        print("-" * 50)
        
        config = extract_model_config(file)
        configs[file] = config
        
        if config:
            for key, value in config.items():
                print(f"  {key}: {value}")
        else:
            print("  No configuration found")
    
    # Compare configurations
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    
    # Group by maze number
    maze_groups = defaultdict(list)
    for file, config in configs.items():
        maze_num = re.search(r'maze(\d+)', file)
        if maze_num:
            maze_groups[maze_num.group(1)].append((file, config))
    
    for maze_num in sorted(maze_groups.keys()):
        print(f"\nMaze {maze_num} Models:")
        print("-" * 30)
        
        files = maze_groups[maze_num]
        if len(files) == 2:  # Should have baseline and FMSL
            baseline_file, baseline_config = files[0]
            fmsl_file, fmsl_config = files[1]
            
            print(f"  Baseline: {baseline_file}")
            print(f"  FMSL:     {fmsl_file}")
            
            # Compare key parameters
            key_params = ['filts', 'nb_fc_node', 'nb_classes', 'sample_rate', 'first_conv']
            
            print("  Comparison:")
            for param in key_params:
                baseline_val = baseline_config.get(param, 'N/A')
                fmsl_val = fmsl_config.get(param, 'N/A')
                
                if baseline_val == fmsl_val:
                    status = "✓ MATCH"
                else:
                    status = "✗ DIFFERENT"
                
                print(f"    {param}: {baseline_val} vs {fmsl_val} {status}")
        else:
            print(f"  Found {len(files)} files (expected 2)")
            for file, config in files:
                print(f"    - {file}")

def create_standardized_config():
    """Create a standardized configuration file"""
    
    standardized_config = {
        'base_architecture': {
            'filts': [128, [128, 128], [128, 256]],
            'nb_fc_node': 1024,
            'nb_classes': 2,
            'sample_rate': 16000,
            'first_conv': 251,
            'dropout_rate': 0.3
        },
        'wav2vec2_config': {
            'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
            'wav2vec2_output_dim': 768,
            'wav2vec2_freeze': True
        },
        'fmsl_config': {
            'fmsl_type': 'prototype',
            'fmsl_n_prototypes': 3,
            'fmsl_s': 32.0,
            'fmsl_m': 0.45,
            'fmsl_enable_lsa': False
        },
        'training_config': {
            'batch_size': 12,
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'grad_clip_norm': 1.0,
            'num_epochs': 5,
            'seed': 1234
        }
    }
    
    # Write standardized config
    with open('standardized_maze_config.py', 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('Standardized configuration for all maze models\n')
        f.write('This ensures architectural consistency across all models\n')
        f.write('"""\n\n')
        f.write('STANDARDIZED_CONFIG = {\n')
        
        for category, params in standardized_config.items():
            f.write(f"    '{category}': {{\n")
            for key, value in params.items():
                if isinstance(value, str):
                    f.write(f"        '{key}': '{value}',\n")
                else:
                    f.write(f"        '{key}': {value},\n")
            f.write('    },\n')
        
        f.write('}\n\n')
        f.write('def get_standardized_config(maze_type="baseline"):\n')
        f.write('    """Get standardized configuration for maze models"""\n')
        f.write('    config = STANDARDIZED_CONFIG.copy()\n')
        f.write('    \n')
        f.write('    if maze_type == "fmsl":\n')
        f.write('        # Add FMSL-specific parameters\n')
        f.write('        config.update(STANDARDIZED_CONFIG["fmsl_config"])\n')
        f.write('    \n')
        f.write('    return config\n')
    
    print("\nCreated standardized_maze_config.py")

if __name__ == "__main__":
    analyze_all_mazes()
    create_standardized_config()