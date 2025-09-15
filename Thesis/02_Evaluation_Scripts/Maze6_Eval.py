#!/usr/bin/env python3
# ===================================================================
# Maze6_Eval.py - Comprehensive Evaluation for MAZE6 and MAZE6 FMSL Models
# 
# DESCRIPTION:
# This script provides comprehensive evaluation for both MAZE6 and MAZE6 FMSL models.
# It's designed to work in Google Colab with all dependencies pre-installed.
# 
# FEATURES:
# - Model structure analysis
# - Support for both MAZE6 and MAZE6 FMSL models
# - Score.txt file generation for both variants
# - Manual model path input (no automation)
# - Colab-compatible paths and structure
# - Comprehensive evaluation metrics
#
# USAGE EXAMPLES:
# 1. Evaluate single model:
#    python Maze6_Eval.py --model_type maze6 --model_path /path/to/maze6_model.pth
#    python Maze6_Eval.py --model_type maze6_fmsl --model_path /path/to/maze6_fmsl_model.pth
#
# 2. Evaluate both models and generate score files:
#    python Maze6_Eval.py --run_both_models --maze6_model_path /path/to/maze6_model.pth --maze6_fmsl_model_path /path/to/maze6_fmsl_model.pth
#
# 3. Analysis only (no evaluation):
#    python Maze6_Eval.py --model_type maze6 --model_path /path/to/model.pth --analyze_only
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

# ===================================================================
# Part 1: Model Analysis Functions
# ===================================================================

def analyze_model_structure(model, model_name):
    """Analyze model structure and return detailed information"""
    print(f"\n{'='*60}")
    print(f"MODEL ANALYSIS: {model_name}")
    print(f"{'='*60}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Analyze model components
    components = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            module_type = type(module).__name__
            if module_type not in components:
                components[module_type] = 0
            components[module_type] += 1
    
    print(f"\nModel Components:")
    for comp_type, count in sorted(components.items()):
        print(f"  {comp_type}: {count}")
    
    # Check for FMSL components
    has_fmsl = any('fmsl' in name.lower() for name, _ in model.named_modules())
    print(f"\nFMSL Integration: {'YES' if has_fmsl else 'NO'}")
    
    if has_fmsl:
        fmsl_params = sum(p.numel() for name, p in model.named_parameters() if 'fmsl' in name.lower())
        print(f"FMSL parameters: {fmsl_params:,} ({fmsl_params/1e6:.2f}M)")
    
    # Check for Wav2Vec2 components
    has_wav2vec2 = any('wav2vec2' in name.lower() for name, _ in model.named_modules())
    print(f"Wav2Vec2 Integration: {'YES' if has_wav2vec2 else 'NO'}")
    
    if has_wav2vec2:
        wav2vec2_params = sum(p.numel() for name, p in model.named_parameters() if 'wav2vec2' in name.lower())
        print(f"Wav2Vec2 parameters: {wav2vec2_params:,} ({wav2vec2_params/1e6:.2f}M)")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'components': components,
        'has_fmsl': has_fmsl,
        'fmsl_params': fmsl_params if has_fmsl else 0,
        'has_wav2vec2': has_wav2vec2,
        'wav2vec2_params': wav2vec2_params if has_wav2vec2 else 0
    }

def test_model_forward_pass(model, device, model_name):
    """Test model forward pass with dummy data"""
    print(f"\n{'='*60}")
    print(f"FORWARD PASS TEST: {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    
    # Test with different input shapes
    test_cases = [
        (1, 64600),      # Single sample
        (4, 64600),      # Batch of 4
        (8, 64600),      # Batch of 8
    ]
    
    results = {}
    
    for batch_size, seq_len in test_cases:
        try:
            # Create dummy input
            dummy_input = torch.randn(batch_size, seq_len).to(device)
            
            # Forward pass
            with torch.no_grad():
                start_time = datetime.now()
                output = model(dummy_input)
                end_time = datetime.now()
                
            # Handle dictionary output from FMSL models
            if isinstance(output, dict):
                if 'logits' in output:
                    output_tensor = output['logits']
                else:
                    raise ValueError(f"Dictionary output missing 'logits' key. Available keys: {list(output.keys())}")
            else:
                output_tensor = output
                
            # Analyze output
            output_shape = output_tensor.shape
            output_mean = output_tensor.mean().item()
            output_std = output_tensor.std().item()
            inference_time = (end_time - start_time).total_seconds() * 1000  # ms
            
            print(f"Input: {dummy_input.shape} -> Output: {output_shape}")
            print(f"  Mean: {output_mean:.4f}, Std: {output_std:.4f}")
            print(f"  Inference time: {inference_time:.2f}ms")
            
            results[f"batch_{batch_size}"] = {
                'input_shape': dummy_input.shape,
                'output_shape': output_shape,
                'output_mean': output_mean,
                'output_std': output_std,
                'inference_time_ms': inference_time
            }
            
        except Exception as e:
            print(f"ERROR with input shape {dummy_input.shape}: {e}")
            results[f"batch_{batch_size}"] = {'error': str(e)}
    
    return results

def check_model_compatibility(model, model_name, device):
    """Check if model is compatible with evaluation pipeline"""
    print(f"\n{'='*60}")
    print(f"COMPATIBILITY CHECK: {model_name}")
    print(f"{'='*60}")
    
    compatibility = {
        'has_forward_method': hasattr(model, 'forward'),
        'has_eval_method': hasattr(model, 'eval'),
        'has_train_method': hasattr(model, 'train'),
        'output_shape_correct': False,
        'device_compatible': True
    }
    
    # Test forward method
    try:
        # Ensure model is on the correct device
        model = model.to(device)
        dummy_input = torch.randn(1, 64600).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        # Handle dictionary output from FMSL models
        if isinstance(output, dict):
            if 'logits' in output:
                output_tensor = output['logits']
            else:
                raise ValueError(f"Dictionary output missing 'logits' key. Available keys: {list(output.keys())}")
        else:
            output_tensor = output
        
        # Check output shape (should be [batch_size, 2] for binary classification)
        if len(output_tensor.shape) == 2 and output_tensor.shape[1] == 2:
            compatibility['output_shape_correct'] = True
            print("✅ Output shape correct: [batch_size, 2]")
        else:
            print(f"❌ Output shape incorrect: {output_tensor.shape}, expected [batch_size, 2]")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        compatibility['forward_pass_error'] = str(e)
    
    # Check device compatibility
    try:
        # Test moving model to different devices
        original_device = next(model.parameters()).device
        model.cpu()
        if torch.cuda.is_available():
            model.cuda()
        model.to(original_device)  # Return to original device
        
        # Ensure model is on the correct device for the test
        model = model.to(device)
        print("✅ Device compatibility: OK")
    except Exception as e:
        print(f"❌ Device compatibility failed: {e}")
        compatibility['device_compatible'] = False
        compatibility['device_error'] = str(e)
    
    # Print compatibility summary
    print(f"\nCompatibility Summary:")
    for key, value in compatibility.items():
        if key.endswith('_error'):
            continue
        status = "✅" if value else "❌"
        print(f"  {status} {key.replace('_', ' ').title()}")
    
    return compatibility

# ===================================================================
# Part 2: Data Loading and Evaluation Functions
# ===================================================================

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    """Generate spoof list from protocol files"""
    d_meta, file_list = {}, []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()
    for line in l_meta:
        parts = line.strip().split()
        if is_eval:
            file_list.append(parts[0])
        else:
            _, key, _, _, label = parts
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
    return (d_meta, file_list) if not is_eval else file_list

def pad(x, max_len=64600):
    """Pad or truncate audio to fixed length"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (num_repeats))[:max_len]

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
            
            return torch.FloatTensor(audio), torch.LongTensor([label]), file_id
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return zero tensor as fallback
            return torch.zeros(self.cut), torch.LongTensor([0]), file_id

def save_scores_to_file(results, model_name, output_dir):
    """Save individual model scores to separate files for further analysis"""
    scores_file = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_scores.txt')
    
    with open(scores_file, 'w') as f:
        for file_id, score in zip(results['file_ids'], results['scores']):
            f.write(f'{file_id} {score}\n')
    
    print(f"✅ Individual scores saved to: {scores_file}")
    return scores_file

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
            for batch_idx, (batch_x, labels, utt_id) in enumerate(data_loader):
                print(f"Debug: Batch {batch_idx}, batch_x shape: {batch_x.shape}, utt_id: {utt_id[:3] if len(utt_id) > 3 else utt_id}")
                try:
                    batch_size_actual = batch_x.size(0)
                    batch_x = batch_x.to(device)
                    
                    # Forward pass with error handling
                    batch_out = model(batch_x)
                    
                    # Handle dictionary output from FMSL models
                    if isinstance(batch_out, dict):
                        if 'logits' in batch_out:
                            batch_out_tensor = batch_out['logits']
                        else:
                            raise ValueError(f"Dictionary output missing 'logits' key. Available keys: {list(batch_out.keys())}")
                    else:
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
                    print(f"Debug: batch_score shape: {batch_score.shape}, sample scores: {batch_score[:3] if len(batch_score) > 3 else batch_score}")
                    
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
                    fh.flush()  # Ensure data is written immediately
                    
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
                            
                            # Handle dictionary output from FMSL models
                            if isinstance(batch_out_small, dict):
                                if 'logits' in batch_out_small:
                                    batch_out_small_tensor = batch_out_small['logits']
                                else:
                                    raise ValueError(f"Dictionary output missing 'logits' key. Available keys: {list(batch_out_small.keys())}")
                            else:
                                batch_out_small_tensor = batch_out_small
                            
                            batch_score_small = (batch_out_small_tensor[:, 1]).data.cpu().numpy().ravel()
                            
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
    
    # Check if file was created and has content
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"📁 Scores file size: {file_size} bytes")
        if file_size == 0:
            print("⚠️ WARNING: Scores file is empty!")
        else:
            # Count lines in the file
            with open(save_path, 'r') as f:
                line_count = sum(1 for line in f)
            print(f"📊 Scores file contains {line_count} lines")
    else:
        print("❌ ERROR: Scores file was not created!")
    
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
# Part 3: Dual Model Evaluation Function
# ===================================================================

def run_dual_model_evaluation(args):
    """Run evaluation for both Maze6 and Maze6 FMSL models and generate score files"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model configuration
    model_config = {
        'filts': [128, [128, 128], [128, 256]],
        'first_conv': 251,
        'sample_rate': 16000,
        'nb_fc_node': 1024,
        'fc_dropout': 0.3,
        'nb_classes': 2,
        'wav2vec2_model_name': 'facebook/wav2vec2-large-960h',
        'wav2vec2_output_dim': 1024,
        'wav2vec2_freeze_cnn': True,
        'wav2vec2_unfrozen_transformers': 0,
        'wav2vec2_output_layers': [0, 6, 12, 18, 24],
        'transformer_num_layers': 2,
        'transformer_nhead': 8,
        'transformer_dim_feedforward': 1024,
        'transformer_dropout': 0.1,
        'use_spec_augment_w2v2': True,
        'spec_aug_freq_mask_param_w2v2': 10,
        'spec_aug_n_freq_masks_w2v2': 1,
        'spec_aug_time_mask_param_w2v2': 10,
        'spec_aug_n_time_masks_w2v2': 1,
        'use_attention_pooling': True,
        'attention_pooling_dim': 256
    }
    
    # Load evaluation data
    eval_protocol = os.path.join(args.protocols_path, f'ASVspoof2019.{args.track}.cm.eval.trl.txt')
    
    if not os.path.exists(eval_protocol):
        print(f"❌ ERROR: Protocol file not found: {eval_protocol}")
        print("Please check the protocols_path argument.")
        return
    
    eval_set = Dataset_ASVspoof_eval(protocol_file=eval_protocol, data_dir=args.database_path)
    print(f"Evaluation dataset size: {len(eval_set)}")
    
    if len(eval_set) == 0:
        print("❌ No valid samples found for evaluation!")
        return
    
    # Models to evaluate
    models_to_evaluate = [
        {
            'name': 'maze6',
            'path': args.maze6_model_path,
            'model_class': 'Model_Maze6',
            'display_name': 'MAZE6 (RawNet + Wav2Vec2 + Transformer)'
        },
        {
            'name': 'maze6_fmsl',
            'path': args.maze6_fmsl_model_path,
            'model_class': 'Model6_RawNet_Wav2Vec2_FMSL_Standardized',
            'display_name': 'MAZE6 FMSL (RawNet + Wav2Vec2 + Transformer + FMSL)'
        }
    ]
    
    # Evaluate each model
    for model_info in models_to_evaluate:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {model_info['display_name']}")
        print(f"{'='*80}")
        
        try:
            # Initialize model
            if model_info['name'] == 'maze6':
                from maze6 import Model_Maze6
                model = Model_Maze6(model_config, device)
            else:
                from maze6_fmsl_standardized import Model6_RawNet_Wav2Vec2_FMSL_Standardized
                model = Model6_RawNet_Wav2Vec2_FMSL_Standardized(model_config, device)
            
            # Load model weights with auto-detection
            state_dict = torch.load(model_info['path'], map_location=device)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"⚠️ Unexpected keys: {len(unexpected_keys)} keys")
            
            model = model.to(device)
            model.eval()
            
            print(f"✅ Model loaded: {model_info['display_name']}")
            
            # Run evaluation
            all_scores = []
            all_file_ids = []
            all_predictions = []
            all_labels = []
            
            data_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
            
            with torch.no_grad():
                for batch_idx, (audio, labels, file_ids) in enumerate(data_loader):
                    audio = audio.to(device)
                    labels = labels.squeeze().to(device)
                    
                    try:
                        # Forward pass
                        if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                            output = model(audio, training=False)
                        else:
                            output = model(audio)
                        
                        # Handle different output formats
                        if isinstance(output, dict):
                            logits = output['logits']
                        else:
                            logits = output
                        
                        # Get predictions and scores
                        probabilities = F.softmax(logits, dim=1)
                        scores = probabilities[:, 1].cpu().numpy()  # Bonafide probability
                        predictions = torch.argmax(logits, dim=1).cpu().numpy()
                        
                        all_scores.extend(scores)
                        all_file_ids.extend(file_ids)
                        all_predictions.extend(predictions)
                        all_labels.extend(labels.cpu().numpy())
                        
                    except Exception as e:
                        print(f"Error in batch {batch_idx}: {e}")
                        continue
            
            # Create results dictionary for score generation
            evaluation_results = {
                'scores': np.array(all_scores),
                'file_ids': all_file_ids,
                'predictions': np.array(all_predictions),
                'labels': np.array(all_labels)
            }
            
            # Generate score.txt file
            score_file_path = save_scores_to_file(evaluation_results, model_info['display_name'], args.output_dir)
            print(f"✅ Score file generated: {score_file_path}")
            
            # Also generate the standard scores file
            standard_scores_file = os.path.join(args.output_dir, f'{model_info["name"]}_scores.txt')
            produce_evaluation_file(eval_set, model, device, standard_scores_file, args.batch_size)
            print(f"✅ Standard scores file generated: {standard_scores_file}")
            
        except Exception as e:
            print(f"❌ ERROR: Failed to evaluate {model_info['name']}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("DUAL MODEL EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_dir}")
    print("Generated files:")
    print(f"  - maze6_scores.txt (Maze6 baseline scores)")
    print(f"  - maze6_fmsl_scores.txt (Maze6 FMSL scores)")
    print(f"  - maze6_(rawnet_+_wav2vec2_+_transformer)_scores.txt (Detailed Maze6 scores)")
    print(f"  - maze6_fmsl_(rawnet_+_wav2vec2_+_transformer_+_fmsl)_scores.txt (Detailed Maze6 FMSL scores)")

# ===================================================================
# Part 4: Main Evaluation Script
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive MAZE6 Model Evaluation')
    parser.add_argument('--model_type', type=str, required=True, choices=['maze6', 'maze6_fmsl'],
                       help='Type of model to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file (MANUAL INPUT REQUIRED)')
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/',
                       help='Root path of ASVspoof2019 database')
    parser.add_argument('--protocols_path', type=str, 
                       default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/',
                       help='Path to protocol files')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'],
                       help='ASVspoof2019 track')
    parser.add_argument('--batch_size', type=int, default=464,
                       help='Batch size for evaluation')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze model structure, skip evaluation')
    parser.add_argument('--run_both_models', action='store_true',
                       help='Run both maze6 and maze6_fmsl models and generate score files for both')
    parser.add_argument('--maze6_model_path', type=str,
                       help='Path to maze6 model (required when using --run_both_models)')
    parser.add_argument('--maze6_fmsl_model_path', type=str,
                       help='Path to maze6_fmsl model (required when using --run_both_models)')
    
    args = parser.parse_args()
    
    # Handle both models mode
    if args.run_both_models:
        if not args.maze6_model_path or not args.maze6_fmsl_model_path:
            print("❌ ERROR: When using --run_both_models, both --maze6_model_path and --maze6_fmsl_model_path are required")
            sys.exit(1)
        
        # Check if both model files exist
        if not os.path.exists(args.maze6_model_path):
            print(f"❌ ERROR: Maze6 model file not found: {args.maze6_model_path}")
            sys.exit(1)
        if not os.path.exists(args.maze6_fmsl_model_path):
            print(f"❌ ERROR: Maze6 FMSL model file not found: {args.maze6_fmsl_model_path}")
            sys.exit(1)
        
        print(f"{'='*80}")
        print(f"MAZE6 DUAL MODEL EVALUATION SYSTEM")
        print(f"{'='*80}")
        print(f"Maze6 Model Path: {args.maze6_model_path}")
        print(f"Maze6 FMSL Model Path: {args.maze6_fmsl_model_path}")
        print(f"Database Path: {args.database_path}")
        print(f"Track: {args.track}")
        print(f"Batch Size: {args.batch_size}")
        print(f"{'='*80}")
        
        # Run both models
        run_dual_model_evaluation(args)
        return
    
    print(f"{'='*80}")
    print(f"MAZE6 MODEL EVALUATION SYSTEM")
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
        'filts': [128, [128, 128], [128, 256]],
        'first_conv': 251,
        'sample_rate': 16000,
        'nb_fc_node': 1024,
        'fc_dropout': 0.3,
        'nb_classes': 2,
        'wav2vec2_model_name': 'facebook/wav2vec2-large-960h',
        'wav2vec2_output_dim': 1024,
        'wav2vec2_freeze_cnn': True,
        'wav2vec2_unfrozen_transformers': 0,
        'wav2vec2_output_layers': [0, 6, 12, 18, 24],
        'transformer_num_layers': 2,
        'transformer_nhead': 8,
        'transformer_dim_feedforward': 1024,
        'transformer_dropout': 0.1,
        'use_spec_augment_w2v2': True,
        'spec_aug_freq_mask_param_w2v2': 10,
        'spec_aug_n_freq_masks_w2v2': 1,
        'spec_aug_time_mask_param_w2v2': 10,
        'spec_aug_n_time_masks_w2v2': 1,
        'use_attention_pooling': True,
        'attention_pooling_dim': 256
    }
    
    # Initialize model based on type
    print(f"\n{'='*60}")
    print(f"INITIALIZING MODEL: {args.model_type.upper()}")
    print(f"{'='*60}")
    
    try:
        if args.model_type == 'maze6':
            # Import and initialize MAZE6 model
            from maze6 import Model_Maze6
            model = Model_Maze6(model_config, device)
            model_name = "MAZE6 (RawNet + Wav2Vec2 + Transformer)"
        elif args.model_type == 'maze6_fmsl':
            # Import and initialize MAZE6 FMSL model
            from maze6_fmsl_standardized import Model6_RawNet_Wav2Vec2_FMSL_Standardized
            model = Model6_RawNet_Wav2Vec2_FMSL_Standardized(model_config, device)
            model_name = "MAZE6 FMSL (RawNet + Wav2Vec2 + Transformer + FMSL)"
        
        print(f"✅ Model initialized: {model_name}")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize model: {e}")
        print("Make sure all dependencies are installed in your Colab environment.")
        sys.exit(1)
    
    # Load model weights with auto-detection
    print(f"\nLoading model weights from: {args.model_path}")
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        
        # Try to load with strict=False to handle dimension mismatches
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys in checkpoint: {len(missing_keys)} keys")
            if len(missing_keys) <= 5:
                for key in missing_keys[:5]:
                    print(f"  - {key}")
            else:
                for key in missing_keys[:3]:
                    print(f"  - {key}")
                print(f"  ... and {len(missing_keys) - 3} more missing keys")
        
        if unexpected_keys:
            print(f"⚠️ Unexpected keys in checkpoint: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys[:5]:
                    print(f"  - {key}")
            else:
                for key in unexpected_keys[:3]:
                    print(f"  - {key}")
                print(f"  ... and {len(unexpected_keys) - 3} more unexpected keys")
        
        print("✅ Model weights loaded successfully (with some mismatches)")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load model weights: {e}")
        print("\nTroubleshooting:")
        print("1. Check if the model file is compatible with the selected model type")
        print("2. For FMSL models, make sure you're using --model_type maze6_fmsl")
        print("3. For regular MAZE6 models, make sure you're using --model_type maze6")
        print("4. The model file might be corrupted or from a different architecture")
        sys.exit(1)
    
    model = model.to(device)
    
    # Model analysis
    analysis_results = analyze_model_structure(model, model_name)
    
    # Forward pass testing
    forward_results = test_model_forward_pass(model, device, model_name)
    
    # Compatibility checking
    compatibility_results = check_model_compatibility(model, model_name, device)
    
    # Save analysis results
    analysis_file = os.path.join(args.output_dir, f'{args.model_type}_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump({
            'model_type': args.model_type,
            'model_name': model_name,
            'model_path': args.model_path,
            'analysis': analysis_results,
            'forward_pass': forward_results,
            'compatibility': compatibility_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✅ Analysis results saved to: {analysis_file}")
    
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
    
    # Create evaluation dataset using the new approach
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
    if args.batch_size > 500:
        print(f"⚠️ Reducing batch size from {args.batch_size} to 32 for CUDA memory management")
        args.batch_size = 32
    
    print(f"📊 Full evaluation dataset: {len(eval_set)} samples")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Estimated batches: {len(eval_set) // args.batch_size + (1 if len(eval_set) % args.batch_size > 0 else 0)}")
    
    # Generate evaluation scores
    scores_file = os.path.join(args.output_dir, f'{args.model_type}_scores.txt')
    produce_evaluation_file(eval_set, model, device, scores_file, args.batch_size)
    
    # Generate detailed results for score.txt creation
    print(f"\n{'='*60}")
    print("GENERATING DETAILED EVALUATION RESULTS")
    print(f"{'='*60}")
    
    # Run evaluation to get detailed results
    model.eval()
    all_scores = []
    all_file_ids = []
    all_predictions = []
    all_labels = []
    
    data_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
    with torch.no_grad():
        for batch_idx, (audio, labels, file_ids) in enumerate(data_loader):
            audio = audio.to(device)
            labels = labels.squeeze().to(device)
            
            try:
                # Forward pass
                if hasattr(model, 'forward') and 'training' in model.forward.__code__.co_varnames:
                    output = model(audio, training=False)
                else:
                    output = model(audio)
                
                # Handle different output formats
                if isinstance(output, dict):
                    logits = output['logits']
                else:
                    logits = output
                
                # Get predictions and scores
                probabilities = F.softmax(logits, dim=1)
                scores = probabilities[:, 1].cpu().numpy()  # Bonafide probability
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_scores.extend(scores)
                all_file_ids.extend(file_ids)
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Create results dictionary for score generation
    evaluation_results = {
        'scores': np.array(all_scores),
        'file_ids': all_file_ids,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }
    
    # Generate score.txt file using the new function
    score_file_path = save_scores_to_file(evaluation_results, model_name, args.output_dir)
    print(f"✅ Score file generated: {score_file_path}")
    
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
    print(f"Analysis Results: {analysis_file}")
    print(f"Scores File: {scores_file}")
    if 'metrics' in locals():
        print(f"Evaluation Results: {results_file}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()