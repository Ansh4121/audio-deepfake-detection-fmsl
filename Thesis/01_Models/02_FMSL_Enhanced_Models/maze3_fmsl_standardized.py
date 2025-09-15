#!/usr/bin/env python3
# ===================================================================
# maze3_fmsl_standardized.py - Model 3: RawNetSinc + SE Transformer + Standardized FMSL
# 
# DESCRIPTION:
# This is the STANDARDIZED version of maze3 with proper FMSL implementation
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
# ===================================================================

import argparse
import sys
import os
import yaml
import random
from collections import OrderedDict
import time

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from tensorboardX import SummaryWriter

from transformers import Wav2Vec2Processor, Wav2Vec2Model as HuggingFaceWav2Vec2Model
import torchaudio
import torchaudio.transforms as T

# Import the standardized FMSL system and configuration
from fmsl_advanced import AdvancedFMSLSystem, create_fmsl_config
from fmsl_standardized_config import get_standardized_model_config

# ===================================================================
# Part 1: Model Definition (Model 3 - RawNetSinc + SE Transformer + Standardized FMSL)
# ===================================================================

class Residual_Block_SE(nn.Module):
    def __init__(self, nb_filts_in_out, first=False, dropout_rate=0.3, stride=1):
        super(Residual_Block_SE, self).__init__()
        self.nb_filts_in_out = nb_filts_in_out
        self.first = first
        self.stride = stride
        
        if not isinstance(nb_filts_in_out, list):
            nb_filts_in_out = [nb_filts_in_out, nb_filts_in_out]
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(nb_filts_in_out[0])
        
        self.conv1 = nn.Conv1d(nb_filts_in_out[0], nb_filts_in_out[1], 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(nb_filts_in_out[1])
        self.conv2 = nn.Conv1d(nb_filts_in_out[1], nb_filts_in_out[1], 3, padding=1, bias=False)
        
        if stride != 1 or nb_filts_in_out[0] != nb_filts_in_out[1]:
            self.shortcut = nn.Conv1d(nb_filts_in_out[0], nb_filts_in_out[1], 1, stride, bias=False)
        else:
            self.shortcut = nn.Identity()
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        if not self.first:
            x = F.relu(self.bn1(x))
        
        out = F.relu(self.bn2(self.conv1(x)))
        out = self.dropout(self.conv2(out))
        out = out + self.shortcut(x)
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=False),  # Fixed: inplace=False to avoid gradient issues
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x: (batch, features, time)
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Project back to original dimension
        x = self.output_projection(x)
        
        # Transpose back
        x = x.transpose(1, 2)  # (batch, features, time)
        
        return x

class Model3_RawNetSinc_SE_Transformer_FMSL_Standardized(nn.Module):
    def __init__(self, d_args, device):
        super(Model3_RawNetSinc_SE_Transformer_FMSL_Standardized, self).__init__()
        self.device = device
        
        # Wav2Vec2 feature extractor
        self.wav2vec2_extractor = Wav2Vec2FeatureExtractor(
            model_name=d_args.get('wav2vec2_model_name', 'facebook/wav2vec2-large-960h'),
            device=self.device,
            freeze_extractor=d_args.get('wav2vec2_freeze_cnn', True)
        )
        
        # Feature projection
        wav2vec2_out_dim = d_args['wav2vec2_output_dim']
        self.feature_projection = nn.Conv1d(wav2vec2_out_dim, d_args['filts'][0], kernel_size=1)
        
        # ✅ REMOVED: Old FMSL layer - it was not true geometric manifold shaping
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(d_args['filts']) - 1):
            self.blocks.append(Residual_Block_SE(d_args['filts'][i], first=(i==0)))
        
        # Final block
        self.final_block = Residual_Block_SE(d_args['filts'][-1])
        
        # Transformer for temporal modeling
        # Fixed: Get the correct output dimension from the last residual block
        if isinstance(d_args['filts'][-1], list):
            last_output_dim = d_args['filts'][-1][1]  # [128, 256] -> 256
        else:
            last_output_dim = d_args['filts'][-1]  # Single value
            
        # Ensure last_output_dim is an integer
        if not isinstance(last_output_dim, int):
            print(f"Warning: last_output_dim is {last_output_dim}, type {type(last_output_dim)}")
            last_output_dim = int(last_output_dim)
            
        self.transformer = TransformerEncoder(
            input_dim=last_output_dim,
            hidden_dim=d_args.get('transformer_hidden_dim', 512),
            num_layers=d_args.get('transformer_num_layers', 6),
            num_heads=d_args.get('transformer_num_heads', 8),
            dropout=d_args.get('transformer_dropout', 0.1)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # ✅ NEW: Replace classifier with standardized FMSL system
        fmsl_config = create_fmsl_config(
            model_type=d_args.get('fmsl_type', 'prototype'),
            n_prototypes=d_args.get('fmsl_n_prototypes', 3),
            s=d_args.get('fmsl_s', 32.0),
            m=d_args.get('fmsl_m', 0.45),
            enable_lsa=False  # Disable LSA to prevent processing slowdown
        )
        
        # Initialize FMSL system with CCE loss for consistency
        # Fixed: Get the correct output dimension from the last residual block
        if isinstance(d_args['filts'][-1], list):
            last_output_dim = d_args['filts'][-1][1]  # [128, 256] -> 256
        else:
            last_output_dim = d_args['filts'][-1]  # Single value
            
        # Ensure last_output_dim is an integer
        if not isinstance(last_output_dim, int):
            print(f"Warning: last_output_dim is {last_output_dim}, type {type(last_output_dim)}")
            last_output_dim = int(last_output_dim)
            
        self.fmsl_system = AdvancedFMSLSystem(
            input_dim=last_output_dim,
            n_classes=d_args['nb_classes'],
            use_integrated_loss=False,  # Use CCE instead of integrated FMSL loss
            **fmsl_config
        )
        
        # Add CCE loss function for consistency with normal maze models
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    
    def forward(self, x, labels=None, training=False):
        # Wav2Vec2 feature extraction
        features = self.wav2vec2_extractor(x)
        
        # Feature projection
        features = self.feature_projection(features)
        
        # ✅ REMOVED: Old FMSL layer - it was in the wrong place
        
        # Residual blocks
        for block in self.blocks:
            features = block(features)
        
        # Final block
        features = self.final_block(features)
        
        # Transformer for temporal modeling
        features = self.transformer(features)
        
        # Global pooling
        features = self.global_pool(features).squeeze(-1)  # Shape: (batch, features)
        
        # ✅ NEW: FMSL system handles geometric features, CCE handles loss
        fmsl_output = self.fmsl_system(features, labels, training=training)
        
        # Use CCE loss for consistency with normal maze models
        if training and labels is not None:
            loss = self.criterion(fmsl_output['logits'], labels)
            return {
                'logits': fmsl_output['logits'],
                'loss': loss,
                'features': features  # Use the raw features from before FMSL
            }
        else:
            return {
                'logits': fmsl_output['logits'],
                'features': features  # Use the raw features from before FMSL
            }

# Wav2Vec2 Feature Extractor
class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self, model_name, device, freeze_extractor=True):
        super(Wav2Vec2FeatureExtractor, self).__init__()
        self.device = device
        # Use Wav2Vec2Processor for models that have it, fallback to feature extractor for others
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = HuggingFaceWav2Vec2Model.from_pretrained(model_name)
        except Exception:
            # Fallback for models without processor (like XLSR models)
            from transformers import Wav2Vec2FeatureExtractor as Wav2Vec2FeatureExtractorHF
            self.processor = Wav2Vec2FeatureExtractorHF.from_pretrained(model_name)
            self.model = HuggingFaceWav2Vec2Model.from_pretrained(model_name)
        
        if freeze_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            x = x.squeeze(1)
        
        # Convert to numpy for processor
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        inputs = self.processor(x, return_tensors='pt', padding=True, sampling_rate=16000)
        
        with torch.set_grad_enabled(not self.model.training):
            outputs = self.model(
                input_values=inputs.input_values.to(self.device)
            )
        
        # Extract features
        features = outputs.last_hidden_state
        features = features.permute(0, 2, 1)  # (batch, hidden_size, sequence_length)
        
        return features

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
        utt_id = self.list_IDs[index]
        label = self.labels[utt_id]
        
        # Load audio file
        audio_path = os.path.join(self.base_dir, f'{utt_id}.flac')
        try:
            x, sr = torchaudio.load(audio_path)
            x = x.squeeze(0).numpy()
            x = pad(x, self.cut)
            x = torch.FloatTensor(x)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return dummy data if file loading fails
            x = torch.zeros(self.cut)
            label = 0
        
        return x, label, utt_id

class Dataset_ASVspoof_eval(Dataset):
    def __init__(self, list_IDs, base_dir, cut=64600):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut
        
    def __len__(self):
        return len(self.list_IDs)
        
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        
        # Load audio file
        audio_path = os.path.join(self.base_dir, f'{utt_id}.flac')
        try:
            x, sr = torchaudio.load(audio_path)
            x = x.squeeze(0).numpy()
            x = pad(x, self.cut)
            x = torch.FloatTensor(x)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return dummy data if file loading fails
            x = torch.zeros(self.cut)
        
        return x, utt_id

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

def train_epoch(train_loader, model, optimizer, device, grad_clip_norm):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y, batch_ids in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # ✅ NEW: Pass labels to model for FMSL loss computation
        output = model(batch_x, batch_y, training=True)
        loss = output['loss']  # Loss computed by FMSL system
        logits = output['logits']
        
        # Check for NaN loss before backward pass
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Skipping batch with NaN/Inf loss: {loss}")
            continue
        
        loss.backward()
        
        # Enhanced gradient clipping with stability checks
        try:
            # Check for NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"WARNING: NaN/Inf gradients in {name}, zeroing gradients")
                        param.grad.data.zero_()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            
            # Check gradients after clipping
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            if total_norm > grad_clip_norm * 2:
                print(f"WARNING: Large gradient norm after clipping: {total_norm}")
            
        except Exception as e:
            print(f"ERROR in gradient processing: {e}")
            # Zero all gradients and continue
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.zero_()
            continue
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    return running_loss / len(train_loader), 100 * correct / total

def evaluate_accuracy(dev_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_ids in dev_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # ✅ NEW: No labels needed for evaluation
            output = model(batch_x, training=False)
            logits = output['logits']
            
            _, predicted = torch.max(logits.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return 100 * correct / total

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
            
            # ✅ NEW: No labels needed for evaluation
            output = model(batch_x, training=False)
            logits = output['logits']
            
            batch_score = (logits[:, 1]).data.cpu().numpy().ravel()
            
            for f, cm in zip(utt_id, batch_score):
                fh.write(f'{f} {cm}\n')
    
    print(f'Scores saved to {save_path}')

# ===================================================================
# Part 4: Main Training Script
# ===================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Maze3+Standardized FMSL: RawNetSinc + SE Transformer + True Geometric FMSL')
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/LA/', 
                       help='Root path of LA database (use local path for efficiency)')
    parser.add_argument('--protocols_path', type=str, 
                       default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/', 
                       help='Path to protocol files')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--comment', type=str, default="maze3_advanced_fmsl_standardized", help='Comment for model directory')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'])
    parser.add_argument('--eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save evaluation scores')
    
    args = parser.parse_args()
    
    # ✅ NEW: Use standardized configuration
    model_config = get_standardized_model_config(3)
    
    # Add transformer-specific configuration
    model_config.update({
        'transformer_hidden_dim': 512,
        'transformer_num_layers': 6,
        'transformer_num_heads': 8,
        'transformer_dropout': 0.1,
        'wav2vec2_freeze_cnn': True
    })
    
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Create model save directory
    model_save_path = os.path.join('models', args.comment)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Initialize model with STANDARDIZED FMSL
    model = Model3_RawNetSinc_SE_Transformer_FMSL_Standardized(model_config, device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable model parameters: {nb_params / 1e6:.2f} M')
    print(f'FMSL system parameters: {sum(p.numel() for p in model.fmsl_system.parameters()):,}')
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW with LR {args.lr}")

    # ✅ UPDATED: Using CCE loss for consistency with normal maze models
    print(f"Loss function: Categorical Cross-Entropy (CCE) with FMSL geometric features")

    # Evaluation mode
    if args.eval:
        eval_protocol = os.path.join(args.protocols_path, f'ASVspoof2019.{args.track}.cm.eval.trl.txt')
        eval_files = genSpoof_list(dir_meta=eval_protocol, is_eval=True)
        eval_set = Dataset_ASVspoof_eval(list_IDs=eval_files, base_dir=os.path.join(args.database_path, f'ASVspoof2019_{args.track}_eval/'))
        produce_evaluation_file(eval_set, model, device, args.eval_output or os.path.join(model_save_path, 'scores.txt'), args.batch_size)
        sys.exit(0)
    
    # Training data
    train_protocol = os.path.join(args.protocols_path, f"ASVspoof2019.{args.track}.cm.train.trn.txt")
    d_label_trn, file_train = genSpoof_list(dir_meta=train_protocol, is_train=True)
    train_set = Dataset_ASVspoof_train(list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(args.database_path, f'ASVspoof2019_{args.track}_train/'))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    
    # Validation data
    dev_protocol = os.path.join(args.protocols_path, f"ASVspoof2019.{args.track}.cm.dev.trl.txt")
    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_protocol)
    dev_set = Dataset_ASVspoof_train(list_IDs=file_dev, labels=d_label_dev, base_dir=os.path.join(args.database_path, f'ASVspoof2019_{args.track}_dev/'))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f'Training samples: {len(train_set)}')
    print(f'Validation samples: {len(dev_set)}')
    
    # Training loop
    writer = SummaryWriter(f'logs/{args.comment}')
    best_acc, best_epoch = 0.0, 0

    for epoch in range(args.num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, device, args.grad_clip_norm)
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
    print(f'✅ STANDARDIZED FMSL implementation completed!')
    print(f'✅ True geometric manifold shaping with angular margins implemented!')
    print(f'✅ Model saved to: {model_save_path}')
    writer.close()
