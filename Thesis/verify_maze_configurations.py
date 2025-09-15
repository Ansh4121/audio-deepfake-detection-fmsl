#!/usr/bin/env python3
"""
Verify maze configurations for architectural consistency
"""

import os
import re
import ast
from collections import defaultdict

def extract_config_from_file(file_path):
    """Extract configuration from a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        config = {}
        
        # Extract key parameters using regex
        patterns = {
            'filts': r"'filts':\s*\[([^\]]+)\]",
            'nb_fc_node': r"'nb_fc_node':\s*(\d+)",
            'nb_classes': r"'nb_classes':\s*(\d+)",
            'sample_rate': r"'sample_rate':\s*(\d+)",
            'first_conv': r"'first_conv':\s*(\d+)",
            'wav2vec2_output_dim': r"'wav2vec2_output_dim':\s*(\d+)",
            'wav2vec2_model_name': r"'wav2vec2_model_name':\s*['\"]([^'\"]+)['\"]",
            'dropout_rate': r"'dropout_rate':\s*([\d.]+)",
            'batch_size': r"'batch_size':\s*(\d+)",
            'lr': r"'lr':\s*([\d.e-]+)",
            'weight_decay': r"'weight_decay':\s*([\d.e-]+)",
            'num_epochs': r"'num_epochs':\s*(\d+)",
            'seed': r"'seed':\s*(\d+)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                try:
                    if key in ['filts']:
                        # Parse filts array
                        config[key] = ast.literal_eval(f'[{value}]')
                    elif key in ['wav2vec2_model_name']:
                        config[key] = value
                    else:
                        config[key] = int(value) if '.' not in value else float(value)
                except:
                    config[key] = value
        
        return config
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def verify_maze_configurations():
    """Verify all maze configurations for consistency"""
    
    print("="*80)
    print("MAZE CONFIGURATION VERIFICATION")
    print("="*80)
    
    # Find all maze files
    maze_files = []
    for file in os.listdir('.'):
        if file.startswith('maze') and file.endswith('.py') and 'eval' not in file:
            maze_files.append(file)
    
    # Group by maze number
    maze_groups = defaultdict(list)
    for file in sorted(maze_files):
        maze_match = re.search(r'maze(\d+)', file)
        if maze_match:
            maze_num = maze_match.group(1)
            maze_groups[maze_num].append(file)
    
    # Expected standardized configuration
    expected_config = {
        'filts': [128, [128, 128], [128, 256]],
        'nb_fc_node': 1024,
        'nb_classes': 2,
        'sample_rate': 16000,
        'first_conv': 251,
        'batch_size': 12,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'num_epochs': 5,
        'seed': 1234
    }
    
    print(f"Expected standardized configuration:")
    for key, value in expected_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("CONFIGURATION VERIFICATION RESULTS")
    print("="*80)
    
    all_consistent = True
    
    for maze_num in sorted(maze_groups.keys()):
        print(f"\nMaze {maze_num}:")
        print("-" * 30)
        
        files = maze_groups[maze_num]
        configs = {}
        
        for file in files:
            config = extract_config_from_file(file)
            configs[file] = config
            
            print(f"  {file}:")
            if config:
                for key, value in config.items():
                    if key in expected_config:
                        expected = expected_config[key]
                        if value == expected:
                            status = "✓"
                        else:
                            status = "✗"
                            all_consistent = False
                        print(f"    {key}: {value} (expected: {expected}) {status}")
                    else:
                        print(f"    {key}: {value}")
            else:
                print("    No configuration found")
                all_consistent = False
        
        # Check consistency between baseline and FMSL
        if len(files) >= 2:
            baseline_files = [f for f in files if 'fmsl' not in f]
            fmsl_files = [f for f in files if 'fmsl' in f]
            
            if baseline_files and fmsl_files:
                baseline_config = configs[baseline_files[0]]
                fmsl_config = configs[fmsl_files[0]]
                
                print(f"  Consistency check (baseline vs FMSL):")
                consistent = True
                for key in expected_config:
                    baseline_val = baseline_config.get(key, 'N/A')
                    fmsl_val = fmsl_config.get(key, 'N/A')
                    
                    if baseline_val == fmsl_val:
                        status = "✓"
                    else:
                        status = "✗"
                        consistent = False
                        all_consistent = False
                    
                    print(f"    {key}: {baseline_val} vs {fmsl_val} {status}")
                
                if not consistent:
                    print(f"    ⚠️  Inconsistency found in Maze {maze_num}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_consistent:
        print("✅ All configurations are consistent with standardized settings")
    else:
        print("❌ Some configurations are inconsistent")
        print("\nRecommendations:")
        print("1. Update inconsistent configurations to match standardized settings")
        print("2. Ensure all maze and maze_fmsl pairs have identical base configurations")
        print("3. Use the standardized configuration file for consistency")
    
    return all_consistent

if __name__ == "__main__":
    print("Verifying maze configurations...")
    is_consistent = verify_maze_configurations()
    
    if not is_consistent:
        print("\n⚠️  Configuration inconsistencies found. Please review and update the models.")
    else:
        print("\n✅ All configurations are consistent!")