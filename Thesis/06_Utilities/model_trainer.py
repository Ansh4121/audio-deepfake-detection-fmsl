#!/usr/bin/env python3
"""
Automated model training utility
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml
import json
from datetime import datetime

def load_training_config(config_file):
    """Load training configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_training(model_name, config, data_path, protocols_path):
    """Run training for a specific model"""
    
    # Determine model file path
    if 'fmsl' in model_name.lower():
        model_file = f"Thesis/01_Models/02_FMSL_Enhanced_Models/{model_name}.py"
    else:
        model_file = f"Thesis/01_Models/01_Baseline_Models/{model_name}.py"
    
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return False
    
    # Build command
    cmd = [
        'python', model_file,
        '--database_path', data_path,
        '--protocols_path', protocols_path,
        '--track', 'LA',
        '--batch_size', str(config.get('batch_size', 32)),
        '--num_epochs', str(config.get('num_epochs', 5)),
        '--lr', str(config.get('lr', 0.0001)),
        '--weight_decay', str(config.get('weight_decay', 0.0001))
    ]
    
    # Add model-specific arguments
    if 'fmsl' in model_name.lower():
        cmd.extend(['--fmsl_type', config.get('fmsl_type', 'prototype')])
        cmd.extend(['--fmsl_n_prototypes', str(config.get('fmsl_n_prototypes', 3))])
    
    print(f"🚀 Training {model_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {model_name} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} training failed:")
        print(f"Error: {e.stderr}")
        return False

def run_evaluation(model_name, model_path, data_path, protocols_path):
    """Run evaluation for a specific model"""
    
    # Determine evaluation file path
    eval_file = f"Thesis/02_Evaluation_Scripts/{model_name}_eval.py"
    
    if not os.path.exists(eval_file):
        print(f"❌ Evaluation file not found: {eval_file}")
        return False
    
    # Build command
    cmd = [
        'python', eval_file,
        '--model_type', model_name,
        '--model_path', model_path,
        '--batch_size', '128'
    ]
    
    print(f"🔍 Evaluating {model_name}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {model_name} evaluation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} evaluation failed:")
        print(f"Error: {e.stderr}")
        return False

def train_all_models(config_file, data_path, protocols_path):
    """Train all models specified in configuration"""
    
    config = load_training_config(config_file)
    models = config.get('models', [])
    
    results = {}
    
    for model_config in models:
        model_name = model_config['name']
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        success = run_training(model_name, model_config, data_path, protocols_path)
        results[model_name] = success
        
        if success:
            print(f"✅ {model_name} completed successfully")
        else:
            print(f"❌ {model_name} failed")
    
    # Print summary
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    successful = sum(results.values())
    total = len(results)
    
    for model, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model}: {status}")
    
    print(f"\nOverall: {successful}/{total} models completed successfully")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Automated model training')
    parser.add_argument('--config', required=True, help='Training configuration file')
    parser.add_argument('--data_path', required=True, help='Data directory path')
    parser.add_argument('--protocols_path', required=True, help='Protocols directory path')
    parser.add_argument('--model', help='Specific model to train (optional)')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    
    args = parser.parse_args()
    
    if args.model:
        # Train specific model
        config = load_training_config(args.config)
        model_config = next((m for m in config.get('models', []) if m['name'] == args.model), None)
        
        if not model_config:
            print(f"❌ Model {args.model} not found in configuration")
            return
        
        success = run_training(args.model, model_config, args.data_path, args.protocols_path)
        
        if success and args.evaluate:
            # Find model checkpoint
            model_path = f"models/{args.model}_best.pth"
            if os.path.exists(model_path):
                run_evaluation(args.model, model_path, args.data_path, args.protocols_path)
    else:
        # Train all models
        results = train_all_models(args.config, args.data_path, args.protocols_path)
        
        if args.evaluate:
            print(f"\n{'='*50}")
            print("RUNNING EVALUATIONS")
            print(f"{'='*50}")
            
            for model, success in results.items():
                if success:
                    model_path = f"models/{model}_best.pth"
                    if os.path.exists(model_path):
                        run_evaluation(model, model_path, args.data_path, args.protocols_path)

if __name__ == "__main__":
    main()
