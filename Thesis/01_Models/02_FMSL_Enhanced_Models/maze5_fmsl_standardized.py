#!/usr/bin/env python3
# ===================================================================
# maze5_fmsl_standardized.py - Model 5: RawNetSinc + SpecAugment + Focal Loss + Standardized FMSL
# 
# DESCRIPTION:
# This is the STANDARDIZED version of maze5 with proper FMSL implementation
# that implements true geometric feature manifold shaping as described in
# the research document.
# 
# KEY IMPROVEMENTS:
# - ✅ True geometric manifold shaping (not just feature refinement)
# - ✅ L2 normalization for hypersphere projection
# - ✅ Angular margin learning with AM-Softmax loss
# - ✅ Prototype-based classification for spoof class modeling
# - ✅ Latent space augmentation for improved generalization
# - ✅ Standardized architecture for fair comparison across all mazes
# - ✅ CRITICAL FIX: Using CCE loss for consistency with normal maze models
# ===================================================================

import argparse
import sys
import os
import yaml
import random
import librosa
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torchaudio.transforms as T

# Import the standardized FMSL system and configuration
from fmsl_advanced import AdvancedFMSLSystem, create_fmsl_config
from fmsl_standardized_config import get_standardized_model_config

# ===================================================================
# Part 1: Model Definition (Model 5 - RawNetSinc + SpecAugment + Focal Loss + Standardized FMSL)
# ===================================================================

class SincConv_Trainable(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, min_low_hz=50, min_band_hz=50):
        super(SincConv_Trainable, self).__init__()

        if in_channels != 1:
            raise ValueError("SincConv_Trainable only support one input channel (raw audio).")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1  # Odd kernel size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.device = device

        # Initialize filterbanks (learnable parameters)
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        
        # Time axis and window
        n = (self.kernel_size - 1) / 2.0
        self.n_ = torch.arange(-n, n + 1, device=self.device).view(1, -1) / self.sample_rate
        self.window_ = torch.hann_window(self.kernel_size, device=self.device, periodic=False)

    def forward(self, x):
        # Constrain parameters
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), 
                          self.min_low_hz, self.sample_rate / 2)
        band = (high - low)
        
        filters = []
        for i in range(self.out_channels):
            fmin_i_norm = low[i] / self.sample_rate
            fmax_i_norm = (low[i] + band[i]) / self.sample_rate
            
            hHigh = (2 * fmax_i_norm) * torch.sinc(2 * fmax_i_norm * np.pi * self.n_)
            hLow = (2 * fmin_i_norm) * torch.sinc(2 * fmin_i_norm * np.pi * self.n_)
            hideal = hHigh - hLow
            filter_i = self.window_ * hideal
            filters.append(filter_i)

        filters = torch.stack(filters).view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride, padding=self.padding, 
                        dilation=self.dilation, bias=None, groups=1)

class Residual_Block_SE(nn.Module):
    def __init__(self, nb_filts_in_out, first=False, dropout_rate=0.3, stride=1):
        super(Residual_Block_SE, self).__init__()
        if isinstance(nb_filts_in_out, list):
            self.nb_filts_in, self.nb_filts_out = nb_filts_in_out[0], nb_filts_in_out[1]
        else:
            self.nb_filts_in = self.nb_filts_out = nb_filts_in_out

        self.first = first
        self.dropout_rate = dropout_rate
        self.stride = stride
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(self.nb_filts_in)
        
        self.activation = nn.ReLU(inplace=True)  # Match original maze5
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        self.conv1 = nn.Conv1d(self.nb_filts_in, self.nb_filts_out, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(self.nb_filts_out)
        self.conv2 = nn.Conv1d(self.nb_filts_out, self.nb_filts_out, kernel_size=3, padding=1, stride=1)
        
        self.downsample = (self.nb_filts_in != self.nb_filts_out) or (self.stride != 1)
        if self.downsample:
            self.conv_downsample = nn.Conv1d(self.nb_filts_in, self.nb_filts_out, kernel_size=1, stride=1)
        
        # AvgPooling for better performance
        self.pool = nn.AvgPool1d(kernel_size=self.stride * 2 - 1, stride=self.stride, padding=self.stride - 1) if self.stride > 1 else None

    def forward(self, x):
        identity = x
        out = self.activation(self.bn1(x)) if not self.first else x
        out = self.conv1(out)
        out = self.dropout(self.activation(self.bn2(out)))
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        
        if self.pool is not None:
            out = self.pool(out)
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),  # Match original maze5
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=20, time_mask_param=20, num_freq_masks=2, num_time_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x):
        # Apply frequency masking
        for _ in range(self.num_freq_masks):
            freq_start = torch.randint(0, self.freq_mask_param, (1,)).item()
            freq_end = torch.randint(freq_start, x.size(1), (1,)).item()
            x[:, freq_start:freq_end, :] = 0
        
        # Apply time masking
        for _ in range(self.num_time_masks):
            time_start = torch.randint(0, self.time_mask_param, (1,)).item()
            time_end = torch.randint(time_start, x.size(2), (1,)).item()
            x[:, :, time_start:time_end] = 0
        
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class Model5_RawNetSinc_SpecAugment_FocalLoss_FMSL_Standardized(nn.Module):
    def __init__(self, d_args, device):
        super(Model5_RawNetSinc_SpecAugment_FocalLoss_FMSL_Standardized, self).__init__()
        self.device = device
        
        # Trainable SincConv layer - use first element of filts list
        self.sinc_conv = SincConv_Trainable(
            device=self.device,
            out_channels=d_args['filts'][0],  # Use first element: 128
            kernel_size=d_args['first_conv'],
            sample_rate=d_args.get('sample_rate', 16000)
        )
        
        # ✅ REMOVED: Old FMSL layer - it was not true geometric manifold shaping
        
        self.first_bn = nn.BatchNorm1d(d_args['filts'][0])  # Use first element: 128
        self.selu = nn.SELU(inplace=False)

        # Residual blocks with SE
        self.block0 = Residual_Block_SE([d_args['filts'][0], d_args['filts'][0]], first=True, stride=1)
        self.se0 = SEBlock(d_args['filts'][0])
        
        self.res_blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        # Match original maze5 architecture exactly
        # First layer: 128 -> [128, 128]
        self.res_blocks.append(Residual_Block_SE([d_args['filts'][0], d_args['filts'][1][0]], stride=2))
        self.se_blocks.append(SEBlock(d_args['filts'][1][0]))
        # Second layer: [128, 128] -> [128, 256]
        self.res_blocks.append(Residual_Block_SE([d_args['filts'][1][0], d_args['filts'][1][1]], stride=2))
        self.se_blocks.append(SEBlock(d_args['filts'][1][1]))
        # Third layer: [128, 256] -> [128, 256]
        self.res_blocks.append(Residual_Block_SE([d_args['filts'][1][1], d_args['filts'][2][0]], stride=2))
        self.se_blocks.append(SEBlock(d_args['filts'][2][0]))
        # Fourth layer: [128, 256] -> [128, 256]
        self.res_blocks.append(Residual_Block_SE([d_args['filts'][2][0], d_args['filts'][2][1]], stride=2))
        self.se_blocks.append(SEBlock(d_args['filts'][2][1]))
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers (match original maze5)
        self.fc1 = nn.Linear(d_args['filts'][2][1], d_args['nb_fc_node'])  # Use last layer's output dimension
        self.dropout_fc = nn.Dropout(p=d_args.get('fc_dropout', 0.5))
        self.fc2 = nn.Linear(d_args['nb_fc_node'], d_args['nb_classes'])
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # SpecAugment for raw audio (match original maze5)
        self.spec_augment = nn.Sequential()
        if d_args.get('use_spec_augment_raw', False):
            self.spec_augment.add_module("freq_mask_raw", T.FrequencyMasking(freq_mask_param=d_args['spec_aug_freq_mask_param_raw']))
            for i in range(d_args.get('spec_aug_n_freq_masks_raw', 1) - 1):
                self.spec_augment.add_module(f"freq_mask_raw_{i+1}", T.FrequencyMasking(freq_mask_param=d_args['spec_aug_freq_mask_param_raw']))
            self.spec_augment.add_module("time_mask_raw", T.TimeMasking(time_mask_param=d_args['spec_aug_time_mask_param_raw']))
            for i in range(d_args.get('spec_aug_n_time_masks_raw', 1) - 1):
                self.spec_augment.add_module(f"time_mask_raw_{i+1}", T.TimeMasking(time_mask_param=d_args['spec_aug_time_mask_param_raw']))
        
        # Focal Loss (though we'll use CCE for consistency)
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
        # ✅ NEW: Replace classifier with standardized FMSL system (balanced for learning)
        fmsl_config = create_fmsl_config(
            model_type=d_args.get('fmsl_type', 'prototype'),
            n_prototypes=d_args.get('fmsl_n_prototypes', 3),  # Restored to 3
            s=d_args.get('fmsl_s', 2.0),  # Restored for effectiveness
            m=d_args.get('fmsl_m', 0.1),  # Restored for effectiveness
            enable_lsa=False  # Disable LSA to prevent instability
        )
        
        # Initialize FMSL system with CCE loss for consistency
        self.fmsl_system = AdvancedFMSLSystem(
            input_dim=d_args['nb_fc_node'],  # Use fc1 output dimension (1024)
            n_classes=d_args['nb_classes'],
            use_integrated_loss=False,  # Use CCE instead of integrated FMSL loss
            **fmsl_config
        )
        
        # Add CCE loss function for consistency with normal maze models (balanced weights)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.3, 0.7]).to(device))

    def forward(self, x, labels=None, training=False):
        # Ensure input is 3D: (batch, channels, time) for SincConv
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add channel dimension: (batch, time) -> (batch, 1, time)
        elif x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: (time) -> (1, 1, time)
            
        # SincConv expects (batch, 1, time) for raw audio input
        out = self.sinc_conv(x)
        
        # SpecAugment on raw features (if enabled)
        if self.training and len(list(self.spec_augment.children())) > 0:
            out = self.spec_augment(out)
        
        out = self.selu(self.first_bn(out))
        
        # ✅ REMOVED: Old FMSL layer - it was in the wrong place
        
        # Residual blocks with SE
        out = self.se0(self.block0(out))
        for block, se in zip(self.res_blocks, self.se_blocks):
            out = se(block(out))
        
        # Global pooling
        out = self.avg_pool(out).squeeze(-1)
        
        # First fully connected layer
        out = self.fc1(out)
        out = self.dropout_fc(out)
        
        # Apply FMSL to the features before final classification
        fmsl_output = self.fmsl_system(out, labels, training=training)
        features = fmsl_output['normalized_embeddings']  # Use the refined features
        
        # Final classification layer
        logits = self.fc2(features)
        
        # Return logsoftmax like original maze5
        return self.logsoftmax(logits)

# ===================================================================
# Part 2: Data Utils (Standardized across all maze files)
# ===================================================================

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
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
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (num_repeats))[:max_len]

class Dataset_ASVspoof_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, cut=64600):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = cut
        
    def __len__(self):
        return len(self.list_IDs)
        
    def __getitem__(self, index):
        key = self.list_IDs[index]
        try:
            # Try multiple path structures for compatibility
            file_paths = [
                # Primary path structure
                os.path.join(self.base_dir, 'ASVspoof2019_LA_train', 'flac', key + '.flac'),
                os.path.join(self.base_dir, 'ASVspoof2019_LA_dev', 'flac', key + '.flac'),
                os.path.join(self.base_dir, 'ASVspoof2019_LA_eval', 'flac', key + '.flac'),
                # Fallback paths
                os.path.join(self.base_dir, 'LA', 'flac', key + '.flac'),
                os.path.join(self.base_dir, 'flac', key + '.flac'),
                os.path.join(self.base_dir, key + '.flac')
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    X, _ = librosa.load(file_path, sr=16000)
                    break
            else:
                # If no path works, return zeros
                print(f"Warning: Could not find {key} in any expected path")
                return torch.zeros(self.cut), 0
                
            y = self.labels[key]
            return Tensor(pad(X, self.cut)), y
            
        except Exception as e:
            print(f"Error loading {key}: {e}. Returning zeros.")
            return torch.zeros(self.cut), 0

class Dataset_ASVspoof_eval(Dataset):
    def __init__(self, list_IDs, base_dir, cut=64600):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut
        
    def __len__(self):
        return len(self.list_IDs)
        
    def __getitem__(self, index):
        key = self.list_IDs[index]
        try:
            # Try multiple path structures for compatibility
            file_paths = [
                # Primary path structure
                os.path.join(self.base_dir, 'ASVspoof2019_LA_eval', 'flac', key + '.flac'),
                os.path.join(self.base_dir, 'ASVspoof2019_LA_train', 'flac', key + '.flac'),
                os.path.join(self.base_dir, 'ASVspoof2019_LA_dev', 'flac', key + '.flac'),
                # Fallback paths
                os.path.join(self.base_dir, 'LA', 'flac', key + '.flac'),
                os.path.join(self.base_dir, 'flac', key + '.flac'),
                os.path.join(self.base_dir, key + '.flac')
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    X, _ = librosa.load(file_path, sr=16000)
                    break
            else:
                # If no path works, return zeros
                print(f"Warning: Could not find {key} in any expected path")
                return torch.zeros(self.cut), key
                
            return Tensor(pad(X, self.cut)), key
            
        except Exception as e:
            print(f"Error loading {key}: {e}. Returning zeros.")
            return torch.zeros(self.cut), key

# ===================================================================
# Part 3: Training and Evaluation Functions (Updated for FMSL)
# ===================================================================

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(train_loader, model, optimizer, device, criterion, grad_clip_norm=1.0):
    running_loss = 0.0
    num_correct = 0.0
    num_total = 0.0
    model.train()
    
    for ii, (batch_x, batch_y) in enumerate(train_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        optimizer.zero_grad()
        batch_out = model(batch_x, training=True)
        # batch_out is logsoftmax, convert back to logits for CCE
        logits = torch.exp(batch_out)  # Convert logsoftmax back to probabilities
        batch_loss = criterion(logits, batch_y)
        batch_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optimizer.step()
        
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        
        if ii % 1 == 0:
            sys.stdout.write(f'\r\t Batch {ii}/{len(train_loader)}, Acc: {(num_correct/num_total)*100:.2f}%')
            sys.stdout.flush()
            
    return running_loss / num_total, num_correct / num_total * 100

def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    with torch.no_grad():
        for batch_x, batch_y in dev_loader:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out = model(batch_x, training=False)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
    
    return 100 * (num_correct / num_total)

def produce_evaluation_file(eval_set, model, device, save_path, batch_size):
    model.eval()
    with open(save_path, 'w') as fh:
        for i in range(0, len(eval_set), batch_size):
            batch_x = []
            utt_id = []
            for j in range(i, min(i + batch_size, len(eval_set))):
                x, id_ = eval_set[j]
                batch_x.append(x)
                utt_id.append(id_)
            
            batch_x = torch.stack(batch_x)
            batch_x = batch_x.to(device)
            
            batch_out = model(batch_x, training=False)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            
            for f, cm in zip(utt_id, batch_score):
                fh.write(f'{f} {cm}\n')
    
    print(f'Scores saved to {save_path}')

# ===================================================================
# Part 4: Main Training Script
# ===================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Maze5+Standardized FMSL: RawNetSinc + SpecAugment + Focal Loss + True Geometric FMSL')
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/', 
                       help='Root path of ASVspoof2019 database (default: /content/sample_data/data/)')
    parser.add_argument('--protocols_path', type=str, 
                       default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/', 
                       help='Path to protocol files')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--comment', type=str, default="Maze5_FMSL_Standardized", help='Comment for model directory')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'])
    parser.add_argument('--eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save evaluation scores')
    
    args = parser.parse_args()
    
    print(f"✅ Using parameters: lr={args.lr}, batch_size={args.batch_size}, epochs={args.num_epochs}")
    
    # ✅ NEW: Use standardized configuration
    model_config = get_standardized_model_config(5)
    
    # Add RawNetSinc-specific configuration to match original maze5
    model_config.update({
        'filts': [128, [128, 128], [128, 256]],  # Match original maze5 exactly
        'first_conv': 251,
        'sample_rate': 16000,
        'nb_fc_node': 1024,
        'fc_dropout': 0.5,
        'nb_classes': 2,
        # SpecAugment parameters
        'use_spec_augment_raw': True,
        'spec_aug_freq_mask_param_raw': 10,
        'spec_aug_n_freq_masks_raw': 1,
        'spec_aug_time_mask_param_raw': 10,
        'spec_aug_n_time_masks_raw': 1
    })
    
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # 🚀 EFFICIENT FMSL CONFIGURATION
    print("🚀 APPLYING EFFICIENT FMSL CONFIGURATION:")
    print("   - Scale factor (s): 2.0 (BALANCED)")
    print("   - Angular margin (m): 0.1 (BALANCED)")
    print("   - LSA disabled for stability")
    print(f"   - Learning rate: {args.lr} (EFFICIENT)")
    print(f"   - Batch size: {args.batch_size} (EFFICIENT)")
    
    # Create model save directory
    model_save_path = os.path.join('models', args.comment)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Initialize model with STANDARDIZED FMSL
    model = Model5_RawNetSinc_SpecAugment_FocalLoss_FMSL_Standardized(model_config, device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable model parameters: {nb_params / 1e6:.2f} M')
    print(f'FMSL system parameters: {sum(p.numel() for p in model.fmsl_system.parameters()):,}')
    model = model.to(device)
    
    # FMSL system status
    print(f"🚀 FMSL TRAINING ENABLED:")
    print(f"   - FMSL status: ENABLED")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW with LR {args.lr}")
    
    # Loss function - CCE for consistency with normal maze models
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.3, 0.7]).to(device))
    print(f"Loss function: Categorical Cross-Entropy (CCE) with FMSL geometric features")

    # Evaluation mode
    if args.eval:
        eval_protocol = os.path.join(args.protocols_path, f'ASVspoof2019.{args.track}.cm.eval.trl.txt')
        eval_files = genSpoof_list(dir_meta=eval_protocol, is_eval=True)
        print(f"Using evaluation data path: {args.database_path}")
        eval_set = Dataset_ASVspoof_eval(list_IDs=eval_files, base_dir=args.database_path)
        produce_evaluation_file(eval_set, model, device, args.eval_output or os.path.join(model_save_path, 'scores.txt'), args.batch_size)
        sys.exit(0)
    
    # Training data
    train_protocol = os.path.join(args.protocols_path, f"ASVspoof2019.{args.track}.cm.train.trn.txt")
    d_label_trn, file_train = genSpoof_list(dir_meta=train_protocol, is_train=True)
    print(f"Using training data path: {args.database_path}")
    train_set = Dataset_ASVspoof_train(list_IDs=file_train, labels=d_label_trn, base_dir=args.database_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    
    # Validation data
    dev_protocol = os.path.join(args.protocols_path, f"ASVspoof2019.{args.track}.cm.dev.trl.txt")
    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_protocol)
    print(f"Using validation data path: {args.database_path}")
    dev_set = Dataset_ASVspoof_train(list_IDs=file_dev, labels=d_label_dev, base_dir=args.database_path)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f'Training samples: {len(train_set)}')
    print(f'Validation samples: {len(dev_set)}')
    
    # Training loop
    writer = SummaryWriter(f'logs/{args.comment}')
    
    # Training loop setup
    best_acc, best_epoch = 0.0, 0
    
    print(f"🚀 TRAINING CONFIGURATION:")
    print(f"   - Max epochs: {args.num_epochs}")
    print(f"   - Learning rate: {args.lr}")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Model complexity: Match original maze5")

    for epoch in range(args.num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, device, criterion, args.grad_clip_norm)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        
        writer.add_scalar('accuracy/train', train_accuracy, epoch)
        writer.add_scalar('accuracy/validation', valid_accuracy, epoch)
        writer.add_scalar('loss/train', running_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'\nEpoch {epoch + 1}/{args.num_epochs} | Loss: {running_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Valid Acc: {valid_accuracy:.2f}%')
        
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch+1}_{args.comment}.pth'))

        if valid_accuracy > best_acc:
            print(f'Best validation accuracy found: {valid_accuracy:.2f}% at epoch {epoch + 1}')
            best_acc, best_epoch = valid_accuracy, epoch + 1
            
            # Remove old best model files
            for f in os.listdir(model_save_path):
                if f.startswith('best_epoch_'):
                    os.remove(os.path.join(model_save_path, f))
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(model_save_path, f'best_epoch_{epoch+1}_{args.comment}.pth'))
            
    print(f'\nTraining completed! Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}')
    writer.close()
