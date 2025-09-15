#!/usr/bin/env python3
"""
📊 Score File Processor for Thesis Analysis
==========================================

This module processes score.txt files from maze model evaluations
and prepares data for comprehensive thesis analysis.

Features:
- Load scores from multiple model evaluations
- Calculate performance metrics (EER, MinDCF, AUC, etc.)
- Prepare data for visualization
- Handle missing files gracefully
- Support for both baseline and FMSL variants

Author: AI Assistant
Purpose: Data preparation for thesis analysis
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ScoreFileProcessor:
    def __init__(self, data_dir=".", protocol_file=None):
        self.data_dir = data_dir
        self.protocol_file = protocol_file
        self.scores_data = {}
        self.labels = {}
        self.performance_metrics = {}
        
        # Expected model files (you can modify these paths)
        self.expected_score_files = {
            # Baseline models
            'main': 'main_scores.txt',
            'maze2': 'maze2_scores.txt', 
            'maze3': 'maze3_scores.txt',
            'maze5': 'maze5_scores.txt',
            'maze6': 'maze6_scores.txt',
            'maze7': 'maze7_scores.txt',
            'maze8': 'maze8_scores.txt',
            
            # FMSL variants
            'main_fmsl': 'main_fmsl_scores.txt',
            'maze2_fmsl': 'maze2_fmsl_scores.txt',
            'maze3_fmsl': 'maze3_fmsl_scores.txt',
            'maze5_fmsl': 'maze5_fmsl_scores.txt',
            'maze6_fmsl': 'maze6_fmsl_scores.txt',
            'maze7_fmsl': 'maze7_fmsl_scores.txt',
            'maze8_fmsl': 'maze8_fmsl_scores.txt'
        }
    
    def discover_score_files(self):
        """Discover available score files in the directory"""
        found_files = {}
        
        print("🔍 Discovering score files...")
        
        # Search for score files
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('_scores.txt') or file == 'scores.txt':
                    full_path = os.path.join(root, file)
                    file_lower = file.lower()
                    
                    # Check for FMSL files first (more specific)
                    if 'fmsl' in file_lower:
                        # Extract model name from FMSL file
                        if 'maze' in file_lower:
                            # Extract maze number
                            import re
                            maze_match = re.search(r'maze(\d+)', file_lower)
                            if maze_match:
                                maze_num = maze_match.group(1)
                                model_name = f'maze{maze_num}_fmsl'
                                found_files[model_name] = full_path
                                print(f"   ✅ Found {model_name}: {full_path}")
                                continue
                        elif 'main' in file_lower:
                            found_files['main_fmsl'] = full_path
                            print(f"   ✅ Found main_fmsl: {full_path}")
                            continue
                    
                    # Check for baseline files
                    elif 'maze' in file_lower and 'fmsl' not in file_lower:
                        # Extract maze number
                        import re
                        maze_match = re.search(r'maze(\d+)', file_lower)
                        if maze_match:
                            maze_num = maze_match.group(1)
                            model_name = f'maze{maze_num}'
                            found_files[model_name] = full_path
                            print(f"   ✅ Found {model_name}: {full_path}")
                            continue
                    elif 'main' in file_lower and 'fmsl' not in file_lower:
                        found_files['main'] = full_path
                        print(f"   ✅ Found main: {full_path}")
                        continue
                    
                    # Generic score file fallback
                    if 'scores.txt' in file:
                        base_name = os.path.basename(os.path.dirname(full_path))
                        found_files[base_name] = full_path
                        print(f"   📁 Found generic: {base_name} -> {full_path}")
        
        if not found_files:
            print("   ⚠️ No score files found. Using sample data for demonstration.")
        
        return found_files
    
    def load_protocol_file(self, protocol_path):
        """Load protocol file with labels"""
        if not protocol_path or not os.path.exists(protocol_path):
            print(f"⚠️ Protocol file not found: {protocol_path}")
            return {}
        
        labels = {}
        print(f"📋 Loading protocol file: {protocol_path}")
        
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    file_id = parts[1]  # Usually the second column
                    label_text = parts[-1]  # Last column is label
                    label = 1 if label_text == 'bonafide' else 0
                    labels[file_id] = label
        
        print(f"   📊 Loaded {len(labels)} labels")
        return labels
    
    def load_scores_file(self, scores_path):
        """Load scores from a scores.txt file"""
        if not os.path.exists(scores_path):
            return {}
        
        scores = {}
        with open(scores_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    utt_id, score = parts
                    try:
                        scores[utt_id] = float(score)
                    except ValueError:
                        continue
        
        return scores
    
    def calculate_metrics(self, scores, labels):
        """Calculate performance metrics from scores and labels"""
        if not scores or not labels:
            return None
        
        # Match scores with labels
        y_true = []
        y_scores = []
        
        for utt_id in scores:
            if utt_id in labels:
                y_true.append(labels[utt_id])
                y_scores.append(scores[utt_id])
        
        if len(y_true) < 10:  # Need minimum samples
            return None
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        try:
            # Calculate ROC curve and EER
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
            eer = fpr[eer_idx]
            eer_threshold = thresholds[eer_idx]
            
            # Calculate AUC
            roc_auc = auc(fpr, tpr)
            
            # Calculate precision-recall
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            avg_precision = auc(recall, precision)
            
            # Calculate accuracy at EER threshold
            predictions = (y_scores > eer_threshold).astype(int)
            accuracy = np.mean(predictions == y_true)
            
            # Calculate MinDCF (simplified)
            min_dcf = min(fnr + fpr)
            
            return {
                'eer': eer,
                'eer_percentage': eer * 100,
                'min_dcf': min_dcf,
                'roc_auc': roc_auc,
                'avg_precision': avg_precision,
                'accuracy': accuracy,
                'n_samples': len(y_true),
                'n_bonafide': np.sum(y_true),
                'n_spoof': len(y_true) - np.sum(y_true)
            }
        
        except Exception as e:
            print(f"   ❌ Error calculating metrics: {e}")
            return None
    
    def process_all_scores(self):
        """Process all discovered score files"""
        print("📊 Processing all score files...")
        
        # Discover score files
        found_files = self.discover_score_files()
        
        # Load protocol if available
        if self.protocol_file:
            self.labels = self.load_protocol_file(self.protocol_file)
        
        # Process each score file
        for model_name, score_path in found_files.items():
            print(f"\n🔍 Processing {model_name}...")
            
            # Load scores
            scores = self.load_scores_file(score_path)
            self.scores_data[model_name] = scores
            
            if scores:
                print(f"   📊 Loaded {len(scores)} scores")
                
                # Calculate metrics if labels available
                if self.labels:
                    metrics = self.calculate_metrics(scores, self.labels)
                    if metrics:
                        self.performance_metrics[model_name] = metrics
                        print(f"   ✅ Calculated metrics - EER: {metrics['eer']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
                    else:
                        print(f"   ⚠️ Could not calculate metrics")
                else:
                    print(f"   ⚠️ No labels available for metric calculation")
            else:
                print(f"   ❌ No scores loaded from {score_path}")
        
        return self.performance_metrics
    
    def export_for_thesis_analysis(self, output_file="processed_performance_data.json"):
        """Export processed data for thesis analysis tool"""
        
        if not self.performance_metrics:
            print("⚠️ No performance metrics available. Using sample data.")
            return None
        
        # Prepare data for thesis analyzer
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'real_score_files',
            'models_processed': list(self.performance_metrics.keys()),
            'performance_data': {}
        }
        
        # Convert metrics to thesis analyzer format
        for model_name, metrics in self.performance_metrics.items():
            export_data['performance_data'][model_name] = {
                'eer': metrics['eer'],
                'min_dcf': metrics['min_dcf'], 
                'accuracy': metrics['accuracy'],
                'roc_auc': metrics['roc_auc'],
                'avg_precision': metrics['avg_precision'],
                'n_samples': metrics['n_samples']
            }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Exported performance data to {output_file}")
        print(f"   Models processed: {len(self.performance_metrics)}")
        
        return export_data
    
    def generate_summary_report(self):
        """Generate a summary report of processed data"""
        if not self.performance_metrics:
            print("❌ No data to summarize")
            return
        
        print("\n" + "="*60)
        print("📊 SCORE FILE PROCESSING SUMMARY")
        print("="*60)
        
        # Separate baseline and FMSL models
        baseline_models = {k: v for k, v in self.performance_metrics.items() if 'fmsl' not in k.lower()}
        fmsl_models = {k: v for k, v in self.performance_metrics.items() if 'fmsl' in k.lower()}
        
        print(f"\n📈 Baseline Models ({len(baseline_models)}):")
        for model, metrics in baseline_models.items():
            print(f"   {model:15} - EER: {metrics['eer']:.4f}, Acc: {metrics['accuracy']:.4f}")
        
        print(f"\n🚀 FMSL Models ({len(fmsl_models)}):")
        for model, metrics in fmsl_models.items():
            print(f"   {model:15} - EER: {metrics['eer']:.4f}, Acc: {metrics['accuracy']:.4f}")
        
        # Calculate improvements
        if baseline_models and fmsl_models:
            print(f"\n📊 FMSL Improvements:")
            for fmsl_model in fmsl_models:
                base_model = fmsl_model.replace('_fmsl', '')
                if base_model in baseline_models:
                    base_eer = baseline_models[base_model]['eer']
                    fmsl_eer = fmsl_models[fmsl_model]['eer']
                    improvement = (base_eer - fmsl_eer) / base_eer * 100
                    print(f"   {base_model} -> {fmsl_model}: {improvement:+.2f}% EER improvement")
        
        print("\n" + "="*60)

def main():
    """Main function for score file processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Score File Processor for Thesis Analysis')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Directory to search for score files')
    parser.add_argument('--protocol_file', type=str, 
                       help='Path to protocol file with labels')
    parser.add_argument('--output_file', type=str, default='processed_performance_data.json',
                       help='Output file for processed data')
    
    args = parser.parse_args()
    
    # Create processor
    processor = ScoreFileProcessor(args.data_dir, args.protocol_file)
    
    # Process all score files
    metrics = processor.process_all_scores()
    
    # Generate summary
    processor.generate_summary_report()
    
    # Export for thesis analysis
    if metrics:
        processor.export_for_thesis_analysis(args.output_file)
        print(f"\n🎯 Next steps:")
        print(f"   1. Use the exported data with comprehensive_thesis_analyzer.py")
        print(f"   2. Run: python comprehensive_thesis_analyzer.py --use_real_data {args.output_file}")
    else:
        print(f"\n⚠️ No valid score files found. The thesis analyzer will use sample data.")

if __name__ == '__main__':
    main()
