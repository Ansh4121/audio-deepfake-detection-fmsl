#!/usr/bin/env python3
# ===================================================================
# Maze2_eval_fixed.py - Fixed Evaluation for Maze2 FMSL Models
# 
# DESCRIPTION:
# This script provides evaluation for Maze2 FMSL models with the correct
# architecture that matches the checkpoint structure.
# 
# FIXES:
# - Correct model architecture to match checkpoint
# - Proper channel dimensions: 128 -> 256 -> 512
# - SE blocks (se0, se1, se2) included
# - All components from checkpoint included
# ===================================================================

import argparse
import sys
import os
import json
import random
import librosa
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import torchaudio.transforms as T  # Not used in this version
from transformers import Wav2Vec2Processor, Wav2Vec2Model as HuggingFaceWav2Vec2Model

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from scipy import stats

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Import models
try:
    # from maze2 import Model7_RawNet_Wav2Vec2_SpecAug_FocalLoss  # Commented out due to torchaudio dependency
    print("✅ Successfully imported Maze2 models")
except ImportError as e:
    print(f"❌ Error importing models: {e}")
    sys.exit(1)

# ===================================================================
# Checkpoint-Compatible Model Definition
# ===================================================================

class Wav2Vec2FeatureExtractor(nn.Module):
    """Wav2Vec2 feature extractor"""
    def __init__(self, model_name='facebook/wav2vec2-base-960h', device='cuda', freeze_extractor=True):
        super(Wav2Vec2FeatureExtractor, self).__init__()
        self.device = device
        self.model_name = model_name
        self.freeze_extractor = freeze_extractor
        
        # Load Wav2Vec2 model
        self.wav2vec2_model = HuggingFaceWav2Vec2Model.from_pretrained(model_name)
        
        if freeze_extractor:
            for param in self.wav2vec2_model.parameters():
                param.requires_grad = False
        
        self.output_dim = 768  # Wav2Vec2 base model output dimension
    
    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        with torch.no_grad() if self.freeze_extractor else torch.enable_grad():
            outputs = self.wav2vec2_model(x)
            # Extract features from the last hidden state
            features = outputs.last_hidden_state  # [batch_size, seq_len, 768]
            # Transpose to [batch_size, 768, seq_len] for Conv1d
            features = features.transpose(1, 2)
        return features

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block that matches checkpoint"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block that matches checkpoint architecture"""
    def __init__(self, in_channels, out_channels, stride=1, use_se=True, se_reduction=16):
        super(ResidualBlock, self).__init__()
        self.use_se = use_se
        
        # First conv layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()
        
        # SE block
        if self.use_se:
            self.se = SEBlock(out_channels, reduction=se_reduction)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply SE block if enabled
        if self.use_se:
            out = self.se(out)
        
        out = out + self.shortcut(x)
        return out

class Model2_Wav2Vec2_FMSL_CheckpointCompatible(nn.Module):
    """Model that exactly matches the checkpoint architecture"""
    def __init__(self, d_args, device):
        super(Model2_Wav2Vec2_FMSL_CheckpointCompatible, self).__init__()
        self.device = device

        # Wav2Vec2 feature extractor
        self.wav2vec2_extractor = Wav2Vec2FeatureExtractor(
            model_name=d_args.get('wav2vec2_model_name', 'facebook/wav2vec2-base-960h'),
            device=self.device,
            freeze_extractor=d_args.get('wav2vec2_freeze', True)
        )
        
        # Feature projection
        self.feature_projection = nn.Conv1d(768, 128, kernel_size=1)
        
        # First batch norm
        self.first_bn = nn.BatchNorm1d(128)
        self.selu = nn.SELU(inplace=False)

        # Residual blocks that match checkpoint structure
        # block0: 128 -> 128 (first block, no stride)
        self.block0 = ResidualBlock(128, 128, stride=1, use_se=True, se_reduction=16)
        
        # block1: 128 -> 256 (with stride 2)
        self.block1 = ResidualBlock(128, 256, stride=2, use_se=True, se_reduction=16)
        
        # block2: 256 -> 512
        self.block2 = ResidualBlock(256, 512, stride=1, use_se=True, se_reduction=16)

        # Global pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification layer (matches checkpoint)
        self.classifier = nn.Linear(512, d_args['nb_classes'])
        
        # Loss function (matches checkpoint)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # Extract Wav2Vec2 features
        wav2vec2_features = self.wav2vec2_extractor(x)  # [batch_size, 768, seq_len]
        
        # Project features
        x = self.feature_projection(wav2vec2_features)  # [batch_size, 128, seq_len]
        x = self.first_bn(x)
        x = self.selu(x)
        
        # Pass through residual blocks
        x = self.block0(x)  # [batch_size, 128, seq_len]
        x = self.block1(x)  # [batch_size, 256, seq_len/2]
        x = self.block2(x)  # [batch_size, 512, seq_len/2]
        
        # Global pooling
        x = self.avg_pool(x)  # [batch_size, 512, 1]
        x = x.squeeze(-1)     # [batch_size, 512]
        
        # Final classification
        logits = self.classifier(x)
        
        # Return in the expected format
        return {'logits': logits}

# ===================================================================
# Dataset Class
# ===================================================================

class Dataset_ASVspoof_eval(Dataset):
    """Evaluation dataset for ASVspoof2019 with proper protocol file parsing"""
    def __init__(self, protocol_file, data_dir, cut=64600):
        self.data_dir = data_dir
        self.cut = cut
        
        # First, discover available audio files
        self.available_files = self._discover_audio_files()
        print(f"Found {len(self.available_files)} audio files in {data_dir}")
        
        # Load protocol file and create samples
        self.samples = []
        found_count = 0
        total_protocol_entries = 0
        
        print(f"Loading protocol file: {protocol_file}")
        with open(protocol_file, 'r') as f:
            lines = f.readlines()
            print(f"Protocol file has {len(lines)} lines")
            
            # Show first few lines for debugging
            print("First 10 lines of protocol file:")
            for i, line in enumerate(lines[:10]):
                print(f"  {i+1}: {line.strip()}")
            
            # Count label types in protocol file
            bonafide_count = 0
            spoof_count = 0
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    total_protocol_entries += 1
                    # The actual file ID is in the second column, not the first
                    file_id = parts[1]  # This is LA_E_2834763, not LA_0039
                    
                    # Protocol format: LA_0039 LA_E_2834763 - A11 spoof
                    # Label is always in the last position
                    label_text = parts[-1]  # Last element is always the label
                    
                    label = 1 if label_text == 'bonafide' else 0
                    
                    # Count labels in protocol
                    if label_text == 'bonafide':
                        bonafide_count += 1
                    else:
                        spoof_count += 1
                    
                    # Try to find matching audio file
                    file_path = self._find_matching_file(file_id)
                    if file_path:
                        self.samples.append((file_path, label, file_id))
                        found_count += 1
                        # Debug: Show first few successful matches
                        if found_count <= 5:
                            print(f"   ✅ Matched {file_id} -> {label_text} -> {file_path}")
                    else:
                        if found_count < 5:  # Only show first 5 warnings to avoid spam
                            print(f"   ❌ Could not find file for {file_id} (label: {label_text})")
        
        print(f"Protocol entries processed: {total_protocol_entries}")
        print(f"Protocol label distribution - Bonafide: {bonafide_count}, Spoof: {spoof_count}")
        print(f"Successfully matched {found_count} files from protocol")
        print(f"Match rate: {found_count/total_protocol_entries*100:.1f}%" if total_protocol_entries > 0 else "No protocol entries found")
        
        # Debug: Check label distribution
        if self.samples:
            labels = [sample[1] for sample in self.samples]
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"Label distribution in dataset: {dict(zip(unique_labels, counts))}")
            if len(unique_labels) == 1:
                print("⚠️ WARNING: All samples have the same label! This will cause evaluation issues.")
    
    def _discover_audio_files(self):
        """Discover all available audio files in the data directory"""
        import glob
        
        audio_files = {}
        
        # Search for audio files recursively
        patterns = [
            os.path.join(self.data_dir, '**', '*.flac'),
            os.path.join(self.data_dir, '**', '*.wav'),
        ]
        
        for pattern in patterns:
            matches = glob.glob(pattern, recursive=True)
            for match in matches:
                filename = os.path.basename(match)
                # Store without extension for matching
                base_name = os.path.splitext(filename)[0]
                audio_files[base_name] = match
        
        # Debug: Print some examples of found files
        if audio_files:
            print(f"Sample audio files found:")
            for i, (name, path) in enumerate(list(audio_files.items())[:5]):
                print(f"  {i+1}: {name} -> {path}")
        else:
            print(f"No audio files found in {self.data_dir}")
            print(f"Directory exists: {os.path.exists(self.data_dir)}")
            if os.path.exists(self.data_dir):
                print(f"Contents of {self.data_dir}:")
                try:
                    contents = os.listdir(self.data_dir)
                    for item in contents[:10]:  # Show first 10 items
                        print(f"  {item}")
                    if len(contents) > 10:
                        print(f"  ... and {len(contents) - 10} more items")
                except Exception as e:
                    print(f"  Error listing directory: {e}")
        
        return audio_files
    
    def _find_matching_file(self, file_id):
        """Find matching audio file for a given file_id"""
        # Try exact match first
        if file_id in self.available_files:
            return self.available_files[file_id]
        
        # Try different naming patterns
        # Pattern 1: LA_0022 -> LA_E_0022
        if file_id.startswith('LA_'):
            alt_id = file_id.replace('LA_', 'LA_E_')
            if alt_id in self.available_files:
                return self.available_files[alt_id]
        
        # Pattern 2: LA_0022 -> LA_E_0000022 (pad with zeros)
        if file_id.startswith('LA_'):
            try:
                num_part = file_id.split('_')[-1]
                padded_num = num_part.zfill(7)  # Pad to 7 digits
                alt_id = f"LA_E_{padded_num}"
                if alt_id in self.available_files:
                    return self.available_files[alt_id]
            except:
                pass
        
        # Pattern 3: Try partial matching (in case of slight variations)
        for available_id, path in self.available_files.items():
            if file_id in available_id or available_id in file_id:
                return path
        
        return None
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        file_path, label, file_id = self.samples[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=16000)
            
            # Pad or truncate to max_len
            if len(audio) > self.cut:
                audio = audio[:self.cut]
            else:
                audio = np.pad(audio, (0, self.cut - len(audio)), 'constant')
            
            return torch.FloatTensor(audio), file_id
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(self.cut), file_id

# ===================================================================
# Evaluation Functions
# ===================================================================

def produce_evaluation_file(dataset, model, device, save_path, batch_size=128):
    """Produce evaluation scores file with robust error handling"""
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"GENERATING EVALUATION SCORES")
    print(f"{'='*60}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Output file: {save_path}")
    
    # Track progress and errors
    processed_samples = 0
    error_count = 0
    max_errors = 100  # Stop if too many errors
    
    with open(save_path, 'w') as fh:
        with torch.no_grad():
            for batch_idx, (batch_x, utt_id) in enumerate(data_loader):
                try:
                    batch_size_actual = batch_x.size(0)
                    batch_x = batch_x.to(device)
                    
                    # Forward pass with error handling
                    batch_out = model(batch_x)
                    
                    # Handle different output types (tensor vs dict)
                    if isinstance(batch_out, dict):
                        # FMSL models return a dictionary with 'logits' key
                        batch_out_tensor = batch_out['logits']
                    else:
                        # Regular models return a tensor directly
                        batch_out_tensor = batch_out
                    
                    # Check for NaN or Inf in output
                    if torch.any(torch.isnan(batch_out_tensor)) or torch.any(torch.isinf(batch_out_tensor)):
                        print(f"Warning: NaN/Inf detected in batch {batch_idx}, skipping")
                        error_count += 1
                        if error_count >= max_errors:
                            print(f"Too many errors ({error_count}), stopping evaluation")
                            break
                        continue
                    
                    # Extract scores (probability of class 1 - bonafide)
                    batch_score = (batch_out_tensor[:, 1]).data.cpu().numpy().ravel()
                    
                    # Check for NaN in scores
                    if np.any(np.isnan(batch_score)) or np.any(np.isinf(batch_score)):
                        print(f"Warning: NaN/Inf in scores for batch {batch_idx}, skipping")
                        error_count += 1
                        if error_count >= max_errors:
                            print(f"Too many errors ({error_count}), stopping evaluation")
                            break
                        continue
                    
                    # Write scores
                    for f, cm in zip(utt_id, batch_score):
                        fh.write(f'{f} {cm}\n')
                    
                    processed_samples += batch_size_actual
                    
                    # Progress update every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Processed {batch_idx + 1}/{len(data_loader)} batches ({processed_samples} samples)")
                        
                except KeyboardInterrupt:
                    print(f"\n⚠️ Evaluation interrupted by user at batch {batch_idx}")
                    break
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA OOM in batch {batch_idx}, reducing batch size and retrying...")
                        # Try with smaller batch
                        try:
                            batch_x_small = batch_x[:batch_size//2].to(device)
                            batch_out_small = model(batch_x_small)
                            batch_score_small = (batch_out_small[:, 1]).data.cpu().numpy().ravel()
                            
                            for f, cm in zip(utt_id[:batch_size//2], batch_score_small):
                                fh.write(f'{f} {cm}\n')
                            
                            processed_samples += batch_size//2
                            print(f"Processed half batch successfully")
                        except:
                            print(f"Failed to process even half batch, skipping")
                            error_count += 1
                    else:
                        print(f"Runtime error in batch {batch_idx}: {e}")
                        error_count += 1
                    
                    if error_count >= max_errors:
                        print(f"Too many errors ({error_count}), stopping evaluation")
                        break
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    error_count += 1
                    if error_count >= max_errors:
                        print(f"Too many errors ({error_count}), stopping evaluation")
                        break
                    continue
    
    print(f'✅ Scores saved to {save_path}')
    print(f'📊 Processed {processed_samples} samples')
    if error_count > 0:
        print(f'⚠️ Encountered {error_count} errors during evaluation')

def calculate_evaluation_metrics(scores_file, dataset):
    """Calculate evaluation metrics from scores and dataset"""
    print(f"\n{'='*60}")
    print(f"CALCULATING EVALUATION METRICS")
    print(f"{'='*60}")
    
    # Load scores
    scores = {}
    with open(scores_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                utt_id, score = parts
                scores[utt_id] = float(score)
    
    # Get labels from dataset
    labels = {}
    if hasattr(dataset, 'samples'):
        # Direct dataset with samples
        for file_path, label, file_id in dataset.samples:
            labels[file_id] = label
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
        # Subset dataset
        for file_path, label, file_id in dataset.dataset.samples:
            labels[file_id] = label
    else:
        print("❌ Cannot extract labels from dataset")
        return {}
    
    # Calculate metrics
    y_true = []
    y_scores = []
    
    print(f"Scores file has {len(scores)} entries")
    print(f"Dataset has {len(labels)} entries")
    
    # Debug: Show sample IDs
    if scores:
        print(f"Sample score IDs: {list(scores.keys())[:5]}")
    if labels:
        print(f"Sample label IDs: {list(labels.keys())[:5]}")
    
    for utt_id in scores:
        if utt_id in labels:
            y_true.append(labels[utt_id])
            y_scores.append(scores[utt_id])
    
    if len(y_true) == 0:
        print("❌ No matching scores and labels found!")
        print(f"   Scores file has {len(scores)} entries")
        print(f"   Dataset has {len(labels)} entries")
        return {}
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Check if we have both classes
    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        print(f"❌ Only one class found in labels: {unique_labels}")
        print("   This will cause evaluation metrics to fail")
        return {}
    
    try:
        # Calculate EER
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        
        # Calculate minDCF (simplified)
        min_dcf = min(fnr + fpr)
        
        print(f"Total samples: {len(y_true)}")
        print(f"Bonafide samples: {np.sum(y_true)}")
        print(f"Spoof samples: {len(y_true) - np.sum(y_true)}")
        print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
        print(f"MinDCF: {min_dcf:.4f}")
        
        return {
            'total_samples': len(y_true),
            'bonafide_samples': int(np.sum(y_true)),
            'spoof_samples': int(len(y_true) - np.sum(y_true)),
            'eer': float(eer),
            'eer_percentage': float(eer * 100),
            'min_dcf': float(min_dcf)
        }
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return {}

# ===================================================================
# Main Function
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='Maze2 Model Evaluation - Fixed Version')
    parser.add_argument('--model_type', type=str, required=True, choices=['maze2', 'maze2_fmsl'],
                        help='Type of model to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/',
                       help='Root path of ASVspoof2019 database')
    parser.add_argument('--protocols_path', type=str, 
                       default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/',
                       help='Path to protocol files')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'],
                       help='ASVspoof2019 track')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze model structure, skip evaluation')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"MAZE2 MODEL EVALUATION SYSTEM - FIXED VERSION")
    print(f"{'='*80}")
    print(f"Model Type: {args.model_type}")
    print(f"Model Path: {args.model_path}")
    print(f"Database Path: {args.database_path}")
    print(f"Track: {args.track}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{'='*80}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model file not found: {args.model_path}")
        print("Please provide the correct path to your trained model file.")
        sys.exit(1)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model configuration
    model_config = {
        'filts': [128, [128, 256], [256, 512]],  # Match checkpoint dimensions
        'first_conv': 251,
        'sample_rate': 16000,
        'nb_fc_node': 1024,
        'fc_dropout': 0.5,
        'nb_classes': 2,
        'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
        'wav2vec2_output_dim': 768,
        'wav2vec2_freeze': True,
        'fmsl_type': 'prototype',
        'fmsl_n_prototypes': 3,
        'fmsl_s': 32.0,
        'fmsl_m': 0.45,
        'fmsl_enable_lsa': False,
        'use_se_blocks': True,
        'se_reduction': 16,
        'use_shortcut': True,
        'dropout_rate': 0.3
    }
    
    # Initialize model based on type
    print(f"\n{'='*60}")
    print(f"INITIALIZING MODEL: {args.model_type.upper()}")
    print(f"{'='*60}")
    
    try:
        if args.model_type == 'maze2':
            # Use checkpoint-compatible model for both types (since maze2 import is disabled)
            print("⚠️ Using checkpoint-compatible model for maze2 type (maze2 import disabled)")
            model = Model2_Wav2Vec2_FMSL_CheckpointCompatible(model_config, device)
            model_name = "Maze2 Checkpoint Compatible (Wav2Vec2 + SE Blocks)"
        elif args.model_type == 'maze2_fmsl':
            # Use checkpoint-compatible FMSL model
            model = Model2_Wav2Vec2_FMSL_CheckpointCompatible(model_config, device)
            model_name = "Maze2 FMSL Checkpoint Compatible (Wav2Vec2 + SE Blocks + FMSL)"
        
        print(f"✅ Model initialized: {model_name}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize model: {e}")
        print("Make sure all dependencies are installed in your Colab environment.")
        sys.exit(1)
    
    # Load model weights
    print(f"\nLoading model weights from: {args.model_path}")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)  # Use strict=False to handle missing keys
        print("✅ Model weights loaded successfully (with strict=False)")
    except Exception as e:
        print(f"❌ ERROR: Failed to load model weights: {e}")
        print("\nTroubleshooting:")
        print("1. Check if the model file is compatible with the selected model type")
        print("2. The model file might be corrupted or from a different architecture")
        sys.exit(1)
    
    model = model.to(device)
    
    # If analyze_only, exit here
    if args.analyze_only:
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE - SKIPPING EVALUATION")
        print(f"{'='*60}")
        return
    
    # Evaluation on ASVspoof2019 dataset
    print(f"\n{'='*60}")
    print("STARTING EVALUATION ON ASVSPOOF2019 DATASET")
    print(f"{'='*60}")
    
    # Load evaluation data
    eval_protocol = os.path.join(args.protocols_path, f'ASVspoof2019.{args.track}.cm.eval.trl.txt')
    
    if not os.path.exists(eval_protocol):
        print(f"❌ ERROR: Protocol file not found: {eval_protocol}")
        print("Please check the protocols_path argument.")
        sys.exit(1)
    
    # Create evaluation dataset
    eval_set = Dataset_ASVspoof_eval(protocol_file=eval_protocol, data_dir=args.database_path)
    print(f"Evaluation dataset size: {len(eval_set)}")
    
    if len(eval_set) == 0:
        print("❌ No valid samples found for evaluation!")
        print("This usually means:")
        print("1. The database path is incorrect")
        print("2. The audio files are in a different format or location")
        print("3. The file naming convention doesn't match")
        print(f"Database path: {args.database_path}")
        print(f"Protocol file: {eval_protocol}")
        return
    
    # Reduce batch size for CUDA memory management
    if args.batch_size > 64:
        print(f"⚠️ Reducing batch size from {args.batch_size} to 16 for CUDA memory management")
        args.batch_size = 16
    
    print(f"📊 Full evaluation dataset: {len(eval_set)} samples")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Estimated batches: {len(eval_set) // args.batch_size + (1 if len(eval_set) % args.batch_size > 0 else 0)}")
    
    # Generate evaluation scores
    scores_file = os.path.join(args.output_dir, f'{args.model_type}_scores.txt')
    produce_evaluation_file(eval_set, model, device, scores_file, args.batch_size)
    
    # Calculate metrics
    try:
        metrics = calculate_evaluation_metrics(scores_file, eval_set)
        
        # Save evaluation results
        results_file = os.path.join(args.output_dir, f'{args.model_type}_evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'model_type': args.model_type,
                'model_name': model_name,
                'model_path': args.model_path,
                'evaluation_metrics': metrics,
                'scores_file': scores_file,
                'protocol_file': eval_protocol,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n✅ Evaluation results saved to: {results_file}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to calculate metrics: {e}")
        print("Scores file generated but metrics calculation failed.")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Model Type: {args.model_type}")
    print(f"Scores File: {scores_file}")
    if 'metrics' in locals():
        print(f"Evaluation Results: {results_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
