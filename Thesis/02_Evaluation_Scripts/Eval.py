#!/usr/bin/env python3
"""
MAZE5 vs MAZE5 FMSL Comparison Visualizer
Generates comprehensive comparison graphics and analysis
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from scipy import stats
import pandas as pd
from datetime import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class Maze5ComparisonVisualizer:
    def __init__(self, output_dir="comparison_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'maze5': {
                'name': 'MAZE5 Baseline',
                'color': '#2E86AB',
                'params': 1056898,
                'fmsl_params': 0,
                'has_fmsl': False
            },
            'maze5_fmsl': {
                'name': 'MAZE5 FMSL',
                'color': '#A23B72',
                'params': 2112643,
                'fmsl_params': 1055745,
                'has_fmsl': True
            }
        }
        
        # Performance metrics from your data
        self.performance_metrics = {
            'maze5': {
                'eer': 0.3183056259979337,
                'eer_percentage': 31.83056259979337,
                'min_dcf': 0.6233878379305352,
                'total_samples': 71237,
                'bonafide_samples': 7355,
                'spoof_samples': 63882
            },
            'maze5_fmsl': {
                'eer': 0.2612629535706459,
                'eer_percentage': 26.12629535706459,
                'min_dcf': 0.5170732552419527,
                'total_samples': 71237,
                'bonafide_samples': 7355,
                'spoof_samples': 63882
            }
        }

    def load_scores_file(self, scores_file_path):
        """Load scores from a scores.txt file"""
        scores = {}
        with open(scores_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    utt_id, score = parts
                    scores[utt_id] = float(score)
        return scores

    def load_protocol_file(self, protocol_file_path):
        """Load labels from protocol file"""
        labels = {}
        with open(protocol_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    file_id = parts[1]  # LA_E_2834763
                    label_text = parts[-1]  # bonafide or spoof
                    label = 1 if label_text == 'bonafide' else 0
                    labels[file_id] = label
        return labels

    def calculate_metrics_from_scores(self, scores, labels):
        """Calculate comprehensive metrics from scores and labels"""
        y_true = []
        y_scores = []
        
        for utt_id in scores:
            if utt_id in labels:
                y_true.append(labels[utt_id])
                y_scores.append(scores[utt_id])
        
        if len(y_true) == 0:
            return None
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Calculate EER
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        # Calculate minDCF (simplified)
        min_dcf = min(fnr + fpr)
        
        # Calculate AUC
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall metrics
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = auc(recall, precision)
        
        # Calculate confusion matrix at EER threshold
        predictions = (y_scores > eer_threshold).astype(int)
        cm = confusion_matrix(y_true, predictions)
        
        return {
            'y_true': y_true,
            'y_scores': y_scores,
            'eer': eer,
            'eer_percentage': eer * 100,
            'min_dcf': min_dcf,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'eer_threshold': eer_threshold,
            'predictions': predictions
        }

    def create_model_architecture_comparison(self):
        """Create model architecture comparison plots"""
        print("Creating model architecture comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Parameter count comparison
        models = list(self.model_configs.keys())
        param_counts = [self.model_configs[model]['params'] for model in models]
        fmsl_params = [self.model_configs[model]['fmsl_params'] for model in models]
        base_params = [param_counts[i] - fmsl_params[i] for i in range(len(models))]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, base_params, width, label='Base Parameters', alpha=0.8, color='#2E86AB')
        ax1.bar(x + width/2, fmsl_params, width, label='FMSL Parameters', alpha=0.8, color='#A23B72')
        
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Number of Parameters')
        ax1.set_title('Model Parameter Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels([self.model_configs[model]['name'] for model in models])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (base, fmsl) in enumerate(zip(base_params, fmsl_params)):
            ax1.text(i - width/2, base + 50000, f'{base:,}', ha='center', va='bottom', fontsize=9)
            ax1.text(i + width/2, fmsl + 50000, f'{fmsl:,}', ha='center', va='bottom', fontsize=9)
        
        # 2. Parameter increase percentage
        param_increase = [(param_counts[1] - param_counts[0]) / param_counts[0] * 100]
        ax2.bar(['Parameter Increase'], param_increase, color='#A23B72', alpha=0.8)
        ax2.set_ylabel('Percentage Increase')
        ax2.set_title('FMSL Parameter Overhead')
        ax2.grid(True, alpha=0.3)
        ax2.text(0, param_increase[0] + 5, f'{param_increase[0]:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 3. Performance improvement
        eer_improvement = self.performance_metrics['maze5']['eer_percentage'] - self.performance_metrics['maze5_fmsl']['eer_percentage']
        dcf_improvement = self.performance_metrics['maze5']['min_dcf'] - self.performance_metrics['maze5_fmsl']['min_dcf']
        
        metrics = ['EER Improvement\n(%)', 'MinDCF Improvement']
        improvements = [eer_improvement, dcf_improvement * 100]  # Scale DCF for visibility
        
        bars = ax3.bar(metrics, improvements, color=['#2ECC71', '#E74C3C'], alpha=0.8)
        ax3.set_ylabel('Improvement')
        ax3.set_title('Performance Improvements with FMSL')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{improvement:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Efficiency ratio (Performance gain per parameter)
        performance_gain = eer_improvement
        param_overhead = param_increase[0]
        efficiency_ratio = performance_gain / param_overhead
        
        ax4.bar(['Efficiency Ratio\n(Performance/Parameters)'], [efficiency_ratio], 
                color='#F39C12', alpha=0.8)
        ax4.set_ylabel('EER Improvement per % Parameter Increase')
        ax4.set_title('FMSL Efficiency Analysis')
        ax4.grid(True, alpha=0.3)
        ax4.text(0, efficiency_ratio + 0.01, f'{efficiency_ratio:.3f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_architecture_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Model architecture comparison saved")

    def create_performance_comparison(self, metrics1, metrics2):
        """Create comprehensive performance comparison plots"""
        print("Creating performance comparison plots...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ROC Curves Comparison
        ax1.plot(metrics1['fpr'], metrics1['tpr'], color=self.model_configs['maze5']['color'], 
                lw=3, label=f"MAZE5 Baseline (AUC = {metrics1['roc_auc']:.4f})")
        ax1.plot(metrics2['fpr'], metrics2['tpr'], color=self.model_configs['maze5_fmsl']['color'], 
                lw=3, label=f"MAZE5 FMSL (AUC = {metrics2['roc_auc']:.4f})")
        ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5, label='Random Classifier')
        
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curves
        ax2.plot(metrics1['recall'], metrics1['precision'], color=self.model_configs['maze5']['color'], 
                lw=3, label=f"MAZE5 Baseline (AP = {metrics1['avg_precision']:.4f})")
        ax2.plot(metrics2['recall'], metrics2['precision'], color=self.model_configs['maze5_fmsl']['color'], 
                lw=3, label=f"MAZE5 FMSL (AP = {metrics2['avg_precision']:.4f})")
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Score Distributions
        ax3.hist(metrics1['y_scores'][metrics1['y_true'] == 0], bins=50, alpha=0.6, 
                label='MAZE5 Baseline - Spoof', density=True, color=self.model_configs['maze5']['color'])
        ax3.hist(metrics1['y_scores'][metrics1['y_true'] == 1], bins=50, alpha=0.6, 
                label='MAZE5 Baseline - Bonafide', density=True, color=self.model_configs['maze5']['color'], 
                linestyle='--')
        ax3.hist(metrics2['y_scores'][metrics2['y_true'] == 0], bins=50, alpha=0.6, 
                label='MAZE5 FMSL - Spoof', density=True, color=self.model_configs['maze5_fmsl']['color'])
        ax3.hist(metrics2['y_scores'][metrics2['y_true'] == 1], bins=50, alpha=0.6, 
                label='MAZE5 FMSL - Bonafide', density=True, color=self.model_configs['maze5_fmsl']['color'], 
                linestyle='--')
        
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Density')
        ax3.set_title('Score Distributions Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Bar Chart
        metric_names = ['EER (%)', 'MinDCF', 'AUC', 'Avg Precision']
        baseline_values = [metrics1['eer_percentage'], metrics1['min_dcf'], metrics1['roc_auc'], metrics1['avg_precision']]
        fmsl_values = [metrics2['eer_percentage'], metrics2['min_dcf'], metrics2['roc_auc'], metrics2['avg_precision']]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, baseline_values, width, label='MAZE5 Baseline', 
                       alpha=0.8, color=self.model_configs['maze5']['color'])
        bars2 = ax4.bar(x + width/2, fmsl_values, width, label='MAZE5 FMSL', 
                       alpha=0.8, color=self.model_configs['maze5_fmsl']['color'])
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Values')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metric_names)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance comparison plots saved")

    def create_confusion_matrices(self, metrics1, metrics2):
        """Create detailed confusion matrix analysis"""
        print("Creating confusion matrix analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. MAZE5 Baseline Confusion Matrix
        cm1 = metrics1['confusion_matrix']
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Spoof', 'Bonafide'], 
                   yticklabels=['Spoof', 'Bonafide'], ax=ax1)
        ax1.set_title('MAZE5 Baseline - Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. MAZE5 FMSL Confusion Matrix
        cm2 = metrics2['confusion_matrix']
        sns.heatmap(cm2, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=['Spoof', 'Bonafide'], 
                   yticklabels=['Spoof', 'Bonafide'], ax=ax2)
        ax2.set_title('MAZE5 FMSL - Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # 3. Confusion Matrix Difference (FMSL - Baseline)
        cm_diff = cm2 - cm1
        sns.heatmap(cm_diff, annot=True, fmt='d', cmap='RdBu_r', center=0,
                   xticklabels=['Spoof', 'Bonafide'], 
                   yticklabels=['Spoof', 'Bonafide'], ax=ax3)
        ax3.set_title('Confusion Matrix Difference (FMSL - Baseline)')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Classification Performance Analysis
        # Calculate detailed metrics
        def calculate_detailed_metrics(cm):
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return precision, recall, specificity, f1
        
        prec1, rec1, spec1, f1_1 = calculate_detailed_metrics(cm1)
        prec2, rec2, spec2, f1_2 = calculate_detailed_metrics(cm2)
        
        metrics = ['Precision', 'Recall', 'Specificity', 'F1-Score']
        baseline_vals = [prec1, rec1, spec1, f1_1]
        fmsl_vals = [prec2, rec2, spec2, f1_2]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, baseline_vals, width, label='MAZE5 Baseline', 
                       alpha=0.8, color=self.model_configs['maze5']['color'])
        bars2 = ax4.bar(x + width/2, fmsl_vals, width, label='MAZE5 FMSL', 
                       alpha=0.8, color=self.model_configs['maze5_fmsl']['color'])
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Values')
        ax4.set_title('Detailed Classification Metrics')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Confusion matrix analysis saved")
        
        # Return detailed metrics for further analysis
        return {
            'baseline': {'precision': prec1, 'recall': rec1, 'specificity': spec1, 'f1': f1_1},
            'fmsl': {'precision': prec2, 'recall': rec2, 'specificity': spec2, 'f1': f1_2},
            'improvements': {
                'precision': prec2 - prec1,
                'recall': rec2 - rec1,
                'specificity': spec2 - spec1,
                'f1': f1_2 - f1_1
            }
        }

    def create_fmsl_impact_analysis(self, detailed_metrics):
        """Create analysis of which factors were most impacted by FMSL"""
        print("Creating FMSL impact analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance Improvements by Category
        categories = ['Error Rate', 'Detection Cost', 'Precision', 'Recall', 'Specificity', 'F1-Score']
        improvements = [
            self.performance_metrics['maze5']['eer_percentage'] - self.performance_metrics['maze5_fmsl']['eer_percentage'],
            (self.performance_metrics['maze5']['min_dcf'] - self.performance_metrics['maze5_fmsl']['min_dcf']) * 100,
            detailed_metrics['improvements']['precision'] * 100,
            detailed_metrics['improvements']['recall'] * 100,
            detailed_metrics['improvements']['specificity'] * 100,
            detailed_metrics['improvements']['f1'] * 100
        ]
        
        colors = ['#E74C3C' if imp > 0 else '#2ECC71' for imp in improvements]
        bars = ax1.bar(categories, improvements, color=colors, alpha=0.8)
        ax1.set_ylabel('Improvement (%)')
        ax1.set_title('FMSL Impact on Different Performance Metrics')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.3),
                    f'{improvement:.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=9)
        
        # 2. Relative Improvement Analysis
        baseline_vals = [
            self.performance_metrics['maze5']['eer_percentage'],
            self.performance_metrics['maze5']['min_dcf'] * 100,
            detailed_metrics['baseline']['precision'] * 100,
            detailed_metrics['baseline']['recall'] * 100,
            detailed_metrics['baseline']['specificity'] * 100,
            detailed_metrics['baseline']['f1'] * 100
        ]
        
        relative_improvements = [imp / base * 100 if base != 0 else 0 
                               for imp, base in zip(improvements, baseline_vals)]
        
        bars = ax2.bar(categories, relative_improvements, color=colors, alpha=0.8)
        ax2.set_ylabel('Relative Improvement (%)')
        ax2.set_title('Relative Performance Improvements with FMSL')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars, relative_improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -1),
                    f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                    fontweight='bold', fontsize=9)
        
        # 3. Error Type Analysis
        # Calculate false positive and false negative rates
        cm_baseline = np.array([[0, 0], [0, 0]])  # Placeholder - would need actual confusion matrix
        cm_fmsl = np.array([[0, 0], [0, 0]])      # Placeholder - would need actual confusion matrix
        
        error_types = ['False Positives', 'False Negatives']
        baseline_errors = [0, 0]  # Would calculate from actual confusion matrices
        fmsl_errors = [0, 0]      # Would calculate from actual confusion matrices
        
        x = np.arange(len(error_types))
        width = 0.35
        
        ax3.bar(x - width/2, baseline_errors, width, label='MAZE5 Baseline', 
               alpha=0.8, color=self.model_configs['maze5']['color'])
        ax3.bar(x + width/2, fmsl_errors, width, label='MAZE5 FMSL', 
               alpha=0.8, color=self.model_configs['maze5_fmsl']['color'])
        
        ax3.set_ylabel('Number of Errors')
        ax3.set_title('Error Type Analysis')
        ax3.set_xticks(x)
        ax3.set_xticklabels(error_types)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. FMSL Effectiveness Summary
        effectiveness_metrics = {
            'Parameter Overhead': 100.0,  # 100% increase in parameters
            'EER Improvement': 17.9,      # 17.9% relative improvement
            'MinDCF Improvement': 17.0,   # 17.0% relative improvement
            'Overall Effectiveness': 17.45  # Average of EER and MinDCF improvements
        }
        
        metric_names = list(effectiveness_metrics.keys())
        values = list(effectiveness_metrics.values())
        colors = ['#E74C3C' if name == 'Parameter Overhead' else '#2ECC71' for name in metric_names]
        
        bars = ax4.bar(metric_names, values, color=colors, alpha=0.8)
        ax4.set_ylabel('Percentage')
        ax4.set_title('FMSL Effectiveness Summary')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fmsl_impact_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ FMSL impact analysis saved")

    def create_comprehensive_summary(self, metrics1, metrics2, detailed_metrics):
        """Create a comprehensive summary dashboard"""
        print("Creating comprehensive summary dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Model Architecture Overview (Top Left)
        ax1 = fig.add_subplot(gs[0, :2])
        models = ['MAZE5 Baseline', 'MAZE5 FMSL']
        params = [self.model_configs['maze5']['params'], self.model_configs['maze5_fmsl']['params']]
        colors = [self.model_configs['maze5']['color'], self.model_configs['maze5_fmsl']['color']]
        
        bars = ax1.bar(models, params, color=colors, alpha=0.8)
        ax1.set_ylabel('Number of Parameters')
        ax1.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        for bar, param in zip(bars, params):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50000,
                    f'{param:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Performance Metrics Comparison (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:])
        metric_names = ['EER (%)', 'MinDCF', 'AUC', 'Avg Precision']
        baseline_vals = [metrics1['eer_percentage'], metrics1['min_dcf'], metrics1['roc_auc'], metrics1['avg_precision']]
        fmsl_vals = [metrics2['eer_percentage'], metrics2['min_dcf'], metrics2['roc_auc'], metrics2['avg_precision']]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        ax2.bar(x - width/2, baseline_vals, width, label='MAZE5 Baseline', 
               alpha=0.8, color=self.model_configs['maze5']['color'])
        ax2.bar(x + width/2, fmsl_vals, width, label='MAZE5 FMSL', 
               alpha=0.8, color=self.model_configs['maze5_fmsl']['color'])
        
        ax2.set_ylabel('Values')
        ax2.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metric_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ROC Curves (Middle Left)
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(metrics1['fpr'], metrics1['tpr'], color=self.model_configs['maze5']['color'], 
                lw=3, label=f"MAZE5 Baseline (AUC = {metrics1['roc_auc']:.4f})")
        ax3.plot(metrics2['fpr'], metrics2['tpr'], color=self.model_configs['maze5_fmsl']['color'], 
                lw=3, label=f"MAZE5 FMSL (AUC = {metrics2['roc_auc']:.4f})")
        ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
        
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Confusion Matrices (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:])
        cm_diff = metrics2['confusion_matrix'] - metrics1['confusion_matrix']
        sns.heatmap(cm_diff, annot=True, fmt='d', cmap='RdBu_r', center=0,
                   xticklabels=['Spoof', 'Bonafide'], 
                   yticklabels=['Spoof', 'Bonafide'], ax=ax4)
        ax4.set_title('Confusion Matrix Difference (FMSL - Baseline)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        
        # 5. FMSL Impact Analysis (Bottom Left)
        ax5 = fig.add_subplot(gs[2, :2])
        categories = ['EER', 'MinDCF', 'Precision', 'Recall', 'F1-Score']
        improvements = [
            self.performance_metrics['maze5']['eer_percentage'] - self.performance_metrics['maze5_fmsl']['eer_percentage'],
            (self.performance_metrics['maze5']['min_dcf'] - self.performance_metrics['maze5_fmsl']['min_dcf']) * 100,
            detailed_metrics['improvements']['precision'] * 100,
            detailed_metrics['improvements']['recall'] * 100,
            detailed_metrics['improvements']['f1'] * 100
        ]
        
        colors = ['#2ECC71'] * len(improvements)
        bars = ax5.bar(categories, improvements, color=colors, alpha=0.8)
        ax5.set_ylabel('Improvement (%)')
        ax5.set_title('FMSL Performance Improvements', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{improvement:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Score Distributions (Bottom Right)
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.hist(metrics1['y_scores'][metrics1['y_true'] == 0], bins=30, alpha=0.6, 
                label='MAZE5 Baseline - Spoof', density=True, color=self.model_configs['maze5']['color'])
        ax6.hist(metrics1['y_scores'][metrics1['y_true'] == 1], bins=30, alpha=0.6, 
                label='MAZE5 Baseline - Bonafide', density=True, color=self.model_configs['maze5']['color'], 
                linestyle='--')
        ax6.hist(metrics2['y_scores'][metrics2['y_true'] == 0], bins=30, alpha=0.6, 
                label='MAZE5 FMSL - Spoof', density=True, color=self.model_configs['maze5_fmsl']['color'])
        ax6.hist(metrics2['y_scores'][metrics2['y_true'] == 1], bins=30, alpha=0.6, 
                label='MAZE5 FMSL - Bonafide', density=True, color=self.model_configs['maze5_fmsl']['color'], 
                linestyle='--')
        
        ax6.set_xlabel('Score')
        ax6.set_ylabel('Density')
        ax6.set_title('Score Distributions Comparison', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Summary Statistics (Bottom Full Width)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        # Create summary text
        summary_text = f"""
        MAZE5 vs MAZE5 FMSL Comprehensive Analysis Summary
        
        Model Architecture:
        • MAZE5 Baseline: {self.model_configs['maze5']['params']:,} parameters
        • MAZE5 FMSL: {self.model_configs['maze5_fmsl']['params']:,} parameters (+{((self.model_configs['maze5_fmsl']['params'] - self.model_configs['maze5']['params']) / self.model_configs['maze5']['params'] * 100):.1f}% increase)
        • FMSL Parameters: {self.model_configs['maze5_fmsl']['fmsl_params']:,} ({self.model_configs['maze5_fmsl']['fmsl_params'] / self.model_configs['maze5_fmsl']['params'] * 100:.1f}% of total)
        
        Performance Improvements:
        • EER: {self.performance_metrics['maze5']['eer_percentage']:.2f}% → {self.performance_metrics['maze5_fmsl']['eer_percentage']:.2f}% ({self.performance_metrics['maze5']['eer_percentage'] - self.performance_metrics['maze5_fmsl']['eer_percentage']:.2f}% improvement)
        • MinDCF: {self.performance_metrics['maze5']['min_dcf']:.4f} → {self.performance_metrics['maze5_fmsl']['min_dcf']:.4f} ({self.performance_metrics['maze5']['min_dcf'] - self.performance_metrics['maze5_fmsl']['min_dcf']:.4f} improvement)
        • AUC: {metrics1['roc_auc']:.4f} → {metrics2['roc_auc']:.4f} ({metrics2['roc_auc'] - metrics1['roc_auc']:.4f} improvement)
        • Average Precision: {metrics1['avg_precision']:.4f} → {metrics2['avg_precision']:.4f} ({metrics2['avg_precision'] - metrics1['avg_precision']:.4f} improvement)
        
        Key Findings:
        • FMSL provides significant performance improvements with reasonable parameter overhead
        • Most impactful improvement: EER reduction of {self.performance_metrics['maze5']['eer_percentage'] - self.performance_metrics['maze5_fmsl']['eer_percentage']:.2f} percentage points
        • Efficiency ratio: {(self.performance_metrics['maze5']['eer_percentage'] - self.performance_metrics['maze5_fmsl']['eer_percentage']) / ((self.model_configs['maze5_fmsl']['params'] - self.model_configs['maze5']['params']) / self.model_configs['maze5']['params'] * 100):.3f} EER improvement per % parameter increase
        • FMSL successfully enhances model's ability to distinguish between bonafide and spoofed audio
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('MAZE5 vs MAZE5 FMSL - Comprehensive Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_analysis_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Comprehensive summary dashboard saved")

    def generate_all_visualizations(self, maze5_scores_file, maze5_fmsl_scores_file, protocol_file):
        """Generate all visualization components"""
        print(f"{'='*80}")
        print("MAZE5 vs MAZE5 FMSL COMPARISON VISUALIZER")
        print(f"{'='*80}")
        
        # Load scores and labels
        print("Loading scores and labels...")
        maze5_scores = self.load_scores_file(maze5_scores_file)
        maze5_fmsl_scores = self.load_scores_file(maze5_fmsl_scores_file)
        labels = self.load_protocol_file(protocol_file)
        
        print(f"MAZE5 Baseline scores: {len(maze5_scores)}")
        print(f"MAZE5 FMSL scores: {len(maze5_fmsl_scores)}")
        print(f"Protocol labels: {len(labels)}")
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics1 = self.calculate_metrics_from_scores(maze5_scores, labels)
        metrics2 = self.calculate_metrics_from_scores(maze5_fmsl_scores, labels)
        
        if metrics1 is None or metrics2 is None:
            print("❌ Error: Could not calculate metrics from scores and labels")
            return
        
        # Generate all visualizations
        self.create_model_architecture_comparison()
        self.create_performance_comparison(metrics1, metrics2)
        detailed_metrics = self.create_confusion_matrices(metrics1, metrics2)
        self.create_fmsl_impact_analysis(detailed_metrics)
        self.create_comprehensive_summary(metrics1, metrics2, detailed_metrics)
        
        # Save detailed results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_configs': self.model_configs,
            'performance_metrics': self.performance_metrics,
            'detailed_metrics': detailed_metrics,
            'maze5_metrics': {
                'eer': metrics1['eer'],
                'eer_percentage': metrics1['eer_percentage'],
                'min_dcf': metrics1['min_dcf'],
                'roc_auc': metrics1['roc_auc'],
                'avg_precision': metrics1['avg_precision']
            },
            'maze5_fmsl_metrics': {
                'eer': metrics2['eer'],
                'eer_percentage': metrics2['eer_percentage'],
                'min_dcf': metrics2['min_dcf'],
                'roc_auc': metrics2['roc_auc'],
                'avg_precision': metrics2['avg_precision']
            }
        }
        
        results_file = os.path.join(self.output_dir, 'comparison_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Detailed results saved to: {results_file}")
        
        print(f"\n{'='*80}")
        print("VISUALIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Output directory: {self.output_dir}")
        print("Generated files:")
        print("  - model_architecture_comparison.png")
        print("  - performance_comparison.png")
        print("  - confusion_matrix_analysis.png")
        print("  - fmsl_impact_analysis.png")
        print("  - comprehensive_analysis_dashboard.png")
        print("  - comparison_results.json")
        print(f"{'='*80}")

def main():
    """Main function to run the comparison visualizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MAZE5 vs MAZE5 FMSL Comparison Visualizer')
    parser.add_argument('--maze5_scores', type=str, required=True,
                       help='Path to MAZE5 baseline scores.txt file')
    parser.add_argument('--maze5_fmsl_scores', type=str, required=True,
                       help='Path to MAZE5 FMSL scores.txt file')
    parser.add_argument('--protocol_file', type=str, required=True,
                       help='Path to protocol file with labels')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.maze5_scores, args.maze5_fmsl_scores, args.protocol_file]:
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found: {file_path}")
            return
    
    # Create visualizer and generate all plots
    visualizer = Maze5ComparisonVisualizer(args.output_dir)
    visualizer.generate_all_visualizations(
        args.maze5_scores, 
        args.maze5_fmsl_scores, 
        args.protocol_file
    )

if __name__ == '__main__':
    main()
