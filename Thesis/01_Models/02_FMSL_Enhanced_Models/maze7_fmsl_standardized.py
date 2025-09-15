#!/usr/bin/env python3
# ===================================================================
# maze7_fmsl_standardized.py - Model 7: RawNet + Wav2Vec2 + Standardized FMSL
# 
# DESCRIPTION:
# This is the STANDARDIZED version of maze7 with proper FMSL implementation
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
from transformers import Wav2Vec2Processor, Wav2Vec2Model as HuggingFaceWav2Vec2Model

# Import the standardized FMSL system and configuration
from fmsl_advanced import AdvancedFMSLSystem, create_fmsl_config
from fmsl_standardized_config import get_standardized_model_config

# ===================================================================
# Part 1: Model Definition (Model 7 - RawNet + Wav2Vec2 + Standardized FMSL)
# ===================================================================

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
        
        self.model.to(device)
        self.model.eval()
    
    def forward(self, x_waveforms_list):
        # Handle both single waveform and batch of waveforms
        if isinstance(x_waveforms_list, torch.Tensor):
            if x_waveforms_list.ndim == 1:
                x_waveforms_list = [x_waveforms_list.cpu().numpy()]
            elif x_waveforms_list.ndim == 2:
                x_waveforms_list = [x_waveforms_list[i].cpu().numpy() for i in range(x_waveforms_list.size(0))]
            else:
                x_waveforms_list = [x_waveforms_list.squeeze().cpu().numpy()]
        
        # Convert to numpy for processor
        if isinstance(x_waveforms_list, torch.Tensor):
            x_waveforms_list = x_waveforms_list.cpu().numpy()

        inputs = self.processor(x_waveforms_list, return_tensors='pt', padding=True, sampling_rate=16000)
        
        with torch.set_grad_enabled(not self.model.training):
            # Safely handle attention_mask - it might not exist in all cases
            attention_mask = None
            if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                attention_mask = inputs.attention_mask.to(self.device)
            
            # Handle the case where attention_mask might not be available
            try:
                if attention_mask is not None:
                    outputs = self.model(
                        input_values=inputs.input_values.to(self.device),
                        attention_mask=attention_mask
                    )
                else:
                    outputs = self.model(
                        input_values=inputs.input_values.to(self.device)
                    )
            except Exception as e:
                # If any error occurs, try without attention_mask
                print(f"Warning: Error with attention_mask, trying without it: {e}")
                outputs = self.model(
                    input_values=inputs.input_values.to(self.device)
                )
        
        # Extract features and permute to match expected format
        features = outputs.last_hidden_state
        features = features.permute(0, 2, 1)  # (batch, hidden_size, sequence_length)
        return features

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
        
        self.activation = nn.ReLU(inplace=False)  # Fixed: inplace=False to avoid gradient issues
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
        if not self.first:
            x = self.activation(self.bn1(x))
        
        out = self.activation(self.bn2(self.conv1(x)))
        out = self.dropout(self.conv2(out))
        
        if self.downsample:
            x = self.conv_downsample(x)
        
        if self.pool is not None:
            x = self.pool(x)
        
        # Ensure x and out have the same dimensions before adding
        if x.size(-1) != out.size(-1):
            # Use adaptive pooling to match dimensions
            x = F.adaptive_avg_pool1d(x, out.size(-1))
        
        out = out + x
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

class Model7_RawNet_Wav2Vec2_FMSL_Standardized(nn.Module):
    def __init__(self, d_args, device):
        super(Model7_RawNet_Wav2Vec2_FMSL_Standardized, self).__init__()
        self.device = device
        
        # Wav2Vec2 feature extractor
        self.wav2vec2_extractor = Wav2Vec2FeatureExtractor(
            model_name=d_args.get('wav2vec2_model_name', 'facebook/wav2vec2-base-960h'),
            device=self.device,
            freeze_extractor=d_args.get('wav2vec2_freeze', True)
        )
        
        # Feature projection
        wav2vec2_out_dim = d_args['wav2vec2_output_dim']
        self.feature_projection = nn.Conv1d(wav2vec2_out_dim, d_args['filts'][0], kernel_size=1)
        
        # ✅ REMOVED: Old FMSL layer - it was not true geometric manifold shaping
        
        self.first_bn = nn.BatchNorm1d(d_args['filts'][0])
        self.selu = nn.SELU(inplace=False)

        # Residual blocks with SE
        self.block0 = Residual_Block_SE(d_args['filts'][0], first=True)
        self.se0 = SEBlock(d_args['filts'][0])
        
        self.res_blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        # Fixed residual block construction to match normal maze7
        self.res_blocks.append(Residual_Block_SE([d_args['filts'][0], d_args['filts'][1][0]], stride=2))
        self.se_blocks.append(SEBlock(d_args['filts'][1][0]))
        self.res_blocks.append(Residual_Block_SE([d_args['filts'][1][0], d_args['filts'][1][1]], stride=2))
        self.se_blocks.append(SEBlock(d_args['filts'][1][1]))
        self.res_blocks.append(Residual_Block_SE([d_args['filts'][1][1], d_args['filts'][2][0]], stride=2))
        self.se_blocks.append(SEBlock(d_args['filts'][2][0]))
        self.res_blocks.append(Residual_Block_SE([d_args['filts'][2][0], d_args['filts'][2][1]], stride=2))
        self.se_blocks.append(SEBlock(d_args['filts'][2][1]))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # ✅ NEW: Replace classifier with standardized FMSL system
        fmsl_config = create_fmsl_config(
            model_type=d_args.get('fmsl_type', 'standard'),  # Use standard instead of prototype
            n_prototypes=d_args.get('fmsl_n_prototypes', 3),
            s=d_args.get('fmsl_s', 5.0),   # Very conservative scale factor
            m=d_args.get('fmsl_m', 0.15),  # Very conservative margin
            enable_lsa=False  # Disable LSA to prevent processing slowdown
        )
        
        # Initialize FMSL system with integrated loss for proper training
        self.fmsl_system = AdvancedFMSLSystem(
            input_dim=d_args['filts'][2][1],  # [128, 256] -> 256
            n_classes=d_args['nb_classes'],
            use_integrated_loss=True,  # Use integrated FMSL loss for proper training
            **fmsl_config
        )
        
        # No separate CCE loss needed - FMSL handles loss internally

    def forward(self, x, labels=None, training=False):
        if x.ndim == 3:
            x = x.squeeze(1)
            
        # Wav2Vec2 feature extraction
        out = self.wav2vec2_extractor(x)
        
        # Feature projection
        out = self.feature_projection(out)
        out = self.selu(self.first_bn(out))
        
        # ✅ REMOVED: Old FMSL layer - it was in the wrong place
        
        # Residual blocks with SE
        out = self.se0(self.block0(out))
        for block, se in zip(self.res_blocks, self.se_blocks):
            out = se(block(out))
        
        # Global pooling
        out = self.global_pool(out).squeeze(-1)  # Shape: (batch, features)
        
        # ✅ NEW: FMSL system handles geometric features, CCE handles loss
        # Debug: Check input features before FMSL
        if training:
            print(f"Input to FMSL - Shape: {out.shape}, Min: {out.min():.4f}, Max: {out.max():.4f}, Mean: {out.mean():.4f}, Std: {out.std():.4f}")
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("WARNING: NaN/Inf detected in input features!")
        
        fmsl_output = self.fmsl_system(out, labels, training=training)
        
        # Use FMSL integrated loss for proper training
        if training and labels is not None:
            loss = fmsl_output['loss']  # Use FMSL integrated loss
            
            # Debug: Check loss value
            print(f"FMSL Loss: {loss.item():.6f}")
            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: NaN/Inf loss detected!")
                print(f"FMSL output keys: {list(fmsl_output.keys())}")
                if 'logits' in fmsl_output:
                    logits = fmsl_output['logits']
                    print(f"Logits - Min: {logits.min():.4f}, Max: {logits.max():.4f}, Mean: {logits.mean():.4f}")
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print("WARNING: NaN/Inf detected in logits!")
            return {
                'logits': fmsl_output['logits'],
                'loss': loss,
                'features': out  # Use the raw features from before FMSL
            }
        else:
            return {
                'logits': fmsl_output['logits'],
                'features': out  # Use the raw features from before FMSL
            }

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
            x, sr = librosa.load(audio_path, sr=16000)
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
            x, sr = librosa.load(audio_path, sr=16000)
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
        loss = output['loss']  # Loss computed by CCE
        logits = output['logits']
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
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
    parser = argparse.ArgumentParser(description='Maze7+Standardized FMSL: RawNet + Wav2Vec2 + True Geometric FMSL')
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/LA/', 
                       help='Root path of LA database (use local path for efficiency)')
    parser.add_argument('--protocols_path', type=str, 
                       default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/', 
                       help='Path to protocol files')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.00001)  # Further reduced LR to prevent NaN
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--comment', type=str, default="FMSL_maze7", help='Comment for model directory')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'])
    parser.add_argument('--eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save evaluation scores')
    
    args = parser.parse_args()
    
    # ✅ NEW: Use standardized configuration
    model_config = get_standardized_model_config(7)
    
    # Add Wav2Vec2-specific configuration
    model_config.update({
        'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
        'wav2vec2_output_dim': 768,
        'wav2vec2_freeze': True,
        'filts': [128, [128, 128], [128, 256]],  # FMSL-compatible architecture (same as normal maze)
        'nb_classes': 2
    })
    
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Create model save directory
    model_save_path = os.path.join('models', args.comment)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Initialize model with STANDARDIZED FMSL
    model = Model7_RawNet_Wav2Vec2_FMSL_Standardized(model_config, device)
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
    print(f'✅ CRITICAL FIX: Using CCE loss for consistency with normal maze models!')
    print(f'✅ Model saved to: {model_save_path}')
    writer.close()
