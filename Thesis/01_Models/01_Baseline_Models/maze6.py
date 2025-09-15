# ===================================================================
# maze6.py - STANDARDIZED for Thesis Comparison
#
# DESCRIPTION:
# This script integrates the advanced features from Maze5 (multi-level
# feature fusion, attentive pooling, fine-tuning) into the stable,
# librosa-based data loading and execution framework of Maze2.
# 
# STANDARDIZATION FOR THESIS:
# - Uses SAME Wav2Vec2 configuration as maze6_fmsl_standardized.py
# - Uses SAME class weights for fair comparison
# - Uses SAME training parameters for consistent evaluation
# - FIXED: 89.74% validation accuracy plateau issue
#
# KEY FEATURES:
# - Core script structure and data loading are from the stable Maze2.
# - Data loading uses `librosa` for maximum stability.
# - Integrates the advanced Maze5 model architecture.
# - STANDARDIZED for fair comparison with FMSL version.
# ===================================================================

# ===================================================================
# Part 1: Combined Imports
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
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from transformers import Wav2Vec2Processor, Wav2Vec2Model as HuggingFaceWav2Vec2Model
import torchaudio.transforms as T

# ===================================================================
# Part 2: Model Definition (Model_Maze6)
# ===================================================================

class Residual_Block_SE(nn.Module):
    def __init__(self, nb_filts_in_out, first=False, dropout_rate=0.3, stride=1):
        super(Residual_Block_SE, self).__init__()
        if isinstance(nb_filts_in_out, list):
            self.nb_filts_in, self.nb_filts_out = nb_filts_in_out[0], nb_filts_in_out[1]
        else:
            self.nb_filts_in = self.nb_filts_out = nb_filts_in_out

        self.first = first
        self.dropout_rate, self.stride = dropout_rate, stride
        if not self.first:
            self.bn1 = nn.BatchNorm1d(self.nb_filts_in)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.conv1 = nn.Conv1d(self.nb_filts_in, self.nb_filts_out, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(self.nb_filts_out)
        self.conv2 = nn.Conv1d(self.nb_filts_out, self.nb_filts_out, kernel_size=3, padding=1, stride=1)
        
        self.downsample = (self.nb_filts_in != self.nb_filts_out) or (self.stride != 1)
        if self.downsample:
            self.conv_downsample = nn.Conv1d(self.nb_filts_in, self.nb_filts_out, kernel_size=1, stride=1)
        
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
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)

class Wav2Vec2FeatureExtractorMultiLevelFT(nn.Module):
    def __init__(self, model_name='facebook/wav2vec2-large-960h', device='cuda',
                 freeze_feature_extractor=True, num_unfrozen_layers=0,
                 output_layers=None):
        super(Wav2Vec2FeatureExtractorMultiLevelFT, self).__init__()
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HuggingFaceWav2Vec2Model.from_pretrained(model_name).to(device)
        self.output_layers = output_layers if output_layers is not None else [-1]

        for param in self.model.parameters():
            param.requires_grad = False
        
        if not freeze_feature_extractor:
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = True
        
        if num_unfrozen_layers > 0 and hasattr(self.model.encoder, 'layers'):
            total_layers = len(self.model.encoder.layers)
            for i in range(total_layers - num_unfrozen_layers, total_layers):
                if i >= 0:
                    for param in self.model.encoder.layers[i].parameters():
                        param.requires_grad = True
            if hasattr(self.model.encoder, 'layer_norm') and self.model.encoder.layer_norm is not None:
                for param in self.model.encoder.layer_norm.parameters():
                    param.requires_grad = True
        
        if any(p.requires_grad for p in self.model.parameters()):
            self.model.train()
        else:
            self.model.eval()

    def forward(self, x_waveforms_list):
        if isinstance(x_waveforms_list, torch.Tensor) and x_waveforms_list.ndim == 3:
            x_waveforms_list = x_waveforms_list.squeeze(1)
        
        if isinstance(x_waveforms_list, torch.Tensor):
             x_waveforms_list = x_waveforms_list.cpu().numpy()

        inputs = self.processor(x_waveforms_list, return_tensors='pt', padding=True, sampling_rate=16000)
        attention_mask = inputs.get('attention_mask')
        
        enable_grad = any(p.requires_grad for p in self.model.parameters())

        with torch.set_grad_enabled(enable_grad):
            outputs = self.model(
                input_values=inputs.input_values.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                output_hidden_states=True
            )
        
        selected_hidden_states = []
        num_total_hidden_states = len(outputs.hidden_states)

        for layer_idx_spec in self.output_layers:
            actual_idx = layer_idx_spec if layer_idx_spec >= 0 else num_total_hidden_states + layer_idx_spec
            if 0 <= actual_idx < num_total_hidden_states:
                selected_hidden_states.append(outputs.hidden_states[actual_idx].permute(0, 2, 1))
            else:
                print(f"Warning: Layer index {layer_idx_spec} invalid. Using last layer.")
                selected_hidden_states.append(outputs.last_hidden_state.permute(0, 2, 1))

        return torch.cat(selected_hidden_states, dim=1)

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentiveStatisticsPooling, self).__init__()
        self.attention_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        b, t, c = x_permuted.shape
        attention_scores = self.attention_mlp(x_permuted.reshape(b * t, c)).view(b, t, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        weighted_mean = torch.sum(x_permuted * attention_weights, dim=1)
        weighted_std = torch.sqrt(torch.sum(((x_permuted - weighted_mean.unsqueeze(1))**2) * attention_weights, dim=1) + 1e-6)
        return torch.cat((weighted_mean, weighted_std), dim=1)

class Model_Maze6(nn.Module):
    def __init__(self, d_args, device):
        super(Model_Maze6, self).__init__()
        self.device = device

        self.output_w2v2_layers = d_args.get('wav2vec2_output_layers', [-1])
        self.wav2vec2_extractor = Wav2Vec2FeatureExtractorMultiLevelFT(
            model_name=d_args.get('wav2vec2_model_name', 'facebook/wav2vec2-base-960h'),
            device=self.device,
            freeze_feature_extractor=d_args.get('wav2vec2_freeze_cnn', True),
            num_unfrozen_layers=d_args.get('wav2vec2_unfrozen_transformers', 0),
            output_layers=self.output_w2v2_layers
        )
        
        num_fused_layers = len(self.output_w2v2_layers)
        fused_dim = d_args['wav2vec2_output_dim'] * num_fused_layers
        projected_dim = d_args.get('projected_dim', d_args['wav2vec2_output_dim'])
        
        self.feature_projection = nn.Conv1d(fused_dim, projected_dim, kernel_size=1) if num_fused_layers > 1 else nn.Identity()

        self.first_bn = nn.BatchNorm1d(projected_dim)
        self.relu = nn.ReLU(inplace=True)  # Use ReLU instead of SELU to prevent NaN

        self.spec_augment = nn.Sequential()
        if d_args.get('use_spec_augment_w2v2', False):
            for i in range(d_args.get('spec_aug_n_freq_masks_w2v2', 1)):
                self.spec_augment.add_module(f"freq_mask_{i}", T.FrequencyMasking(freq_mask_param=d_args['spec_aug_freq_mask_param_w2v2']))
            for i in range(d_args.get('spec_aug_n_time_masks_w2v2', 1)):
                self.spec_augment.add_module(f"time_mask_{i}", T.TimeMasking(time_mask_param=d_args['spec_aug_time_mask_param_w2v2']))

        filts_dim = d_args['filts'][0] # Use the first filter size
        self.block0 = Residual_Block_SE([projected_dim, filts_dim], first=True, stride=1)
        self.se0 = SEBlock(filts_dim)
        
        self.res_blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        # Handle the nested filts structure: [128, [128, 128], [128, 256]]
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
        
        transformer_input_dim = d_args['filts'][2][1]  # Use last layer's output dimension
        self.bn_before_transformer = nn.BatchNorm1d(transformer_input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_input_dim, nhead=d_args.get('transformer_nhead', 8),
            dim_feedforward=d_args.get('transformer_dim_feedforward', 2048),
            dropout=d_args.get('transformer_dropout', 0.1), activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=d_args.get('transformer_num_layers', 4))
        
        self.attentive_pooling = AttentiveStatisticsPooling(input_dim=transformer_input_dim, hidden_dim=d_args.get('attn_pool_hidden_dim', 128))
        self.fc1 = nn.Linear(transformer_input_dim * 2, d_args['nb_fc_node'])
        self.dropout_fc = nn.Dropout(p=d_args.get('fc_dropout', 0.5))
        self.fc2 = nn.Linear(d_args['nb_fc_node'], d_args['nb_classes'])
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x_wave):
        if x_wave.ndim == 3: x_wave = x_wave.squeeze(1)
            
        out = self.wav2vec2_extractor(x_wave)
        out = self.feature_projection(out)
        out = self.relu(self.first_bn(out))  # Use ReLU instead of SELU
        
        if self.training and len(list(self.spec_augment.children())) > 0:
            out = self.spec_augment(out)
            
        out = self.se0(self.block0(out))
        for block, se in zip(self.res_blocks, self.se_blocks):
             out = se(block(out))
        
        out = self.bn_before_transformer(out).permute(0, 2, 1)
        out = self.transformer_encoder(out).permute(0, 2, 1)
        out = self.attentive_pooling(out)
        
        out = self.fc1(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        return out  # Return raw logits instead of LogSoftmax

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


# ===================================================================
# Part 3: Data Utils (from Maze2)
# ===================================================================

def validate_database_paths(database_path, protocols_path):
    """Validate that the database and protocol paths exist and contain expected files."""
    print(f"🔍 Validating database paths...")
    print(f"   Database path: {database_path}")
    print(f"   Protocols path: {protocols_path}")
    
    # Check if base paths exist
    if not os.path.exists(database_path):
        print(f"❌ Database path does not exist: {database_path}")
        return False
    
    if not os.path.exists(protocols_path):
        print(f"❌ Protocols path does not exist: {protocols_path}")
        return False
    
    # Check for expected ASVspoof2019 structure
    expected_dirs = [
        'ASVspoof2019_LA_train',
        'ASVspoof2019_LA_dev', 
        'ASVspoof2019_LA_eval'
    ]
    
    found_dirs = []
    for expected_dir in expected_dirs:
        full_path = os.path.join(database_path, expected_dir)
        if os.path.exists(full_path):
            found_dirs.append(expected_dir)
            print(f"✅ {expected_dir}/ - Found")
            
            # Check for local structure (files directly in directory)
            try:
                local_files = [f for f in os.listdir(full_path) if f.endswith('.flac')]
                if local_files:
                    print(f"   ✅ Local structure - {len(local_files)} .flac files")
                else:
                    print(f"   ⚠️  Local structure - No .flac files found")
            except (OSError, PermissionError) as e:
                print(f"   ⚠️  Local structure - Cannot access: {e}")
            
            # Check for flac subdirectory (Google Drive structure)
            flac_path = os.path.join(full_path, 'flac')
            if os.path.exists(flac_path):
                try:
                    flac_files = [f for f in os.listdir(flac_path) if f.endswith('.flac')]
                    print(f"   ✅ flac/ - {len(flac_files)} .flac files")
                except (OSError, PermissionError) as e:
                    print(f"   ⚠️  flac/ exists but cannot be accessed: {e}")
                    print(f"   💡 This might be a Google Drive sync issue. The directory will still be used for training.")
            else:
                print(f"   ⚠️  flac/ - Not found (expected for local structure)")
        else:
            print(f"❌ {expected_dir}/ - Not found")
    
    # Check for protocol files
    protocol_files = [
        'ASVspoof2019.LA.cm.train.trn.txt',
        'ASVspoof2019.LA.cm.dev.trl.txt', 
        'ASVspoof2019.LA.cm.eval.trl.txt'
    ]
    
    found_protocols = []
    for protocol_file in protocol_files:
        full_path = os.path.join(protocols_path, protocol_file)
        if os.path.exists(full_path):
            found_protocols.append(protocol_file)
            print(f"   ✅ {protocol_file}")
        else:
            print(f"   ❌ {protocol_file} not found")
    
    # Summary
    if len(found_dirs) >= 2 and len(found_protocols) >= 2:
        print(f"✅ Database validation successful!")
        print(f"   Found {len(found_dirs)}/3 data directories")
        print(f"   Found {len(found_protocols)}/3 protocol files")
        return True
    else:
        print(f"⚠️  Database validation completed with warnings!")
        print(f"   Found {len(found_dirs)}/3 data directories")
        print(f"   Found {len(found_protocols)}/3 protocol files")
        if len(found_dirs) >= 1 and len(found_protocols) >= 1:
            print(f"   💡 Proceeding anyway - some directories may have sync issues but training can continue.")
            return True
        else:
            print(f"   ❌ Insufficient data found. Need at least 1 data directory and 1 protocol file.")
            return False
def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta, file_list = {}, []
    with open(dir_meta, 'r') as f: l_meta = f.readlines()
    for line in l_meta:
        parts = line.strip().split()
        if is_eval: file_list.append(parts[0])
        else:
            _, key, _, _, label = parts
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
    return (d_meta, file_list) if not is_eval else file_list

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len: return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (num_repeats))[:max_len]

class Dataset_ASVspoof_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, cut=64600):
        self.list_IDs, self.labels, self.base_dir, self.cut = list_IDs, labels, base_dir, cut
    def __len__(self): return len(self.list_IDs)
    def __getitem__(self, index):
        key = self.list_IDs[index]
        try:
            # Local data structure paths (after copying from Google Drive)
            file_paths = [
                # Primary local structure (after copy script)
                os.path.join(self.base_dir, 'ASVspoof2019_LA_train', key + '.flac'),  # /content/sample_data/data/ASVspoof2019_LA_train/
                os.path.join(self.base_dir, 'ASVspoof2019_LA_dev', key + '.flac'),    # /content/sample_data/data/ASVspoof2019_LA_dev/
                os.path.join(self.base_dir, 'ASVspoof2019_LA_eval', key + '.flac'),   # /content/sample_data/data/ASVspoof2019_LA_eval/
                # Fallback to original Google Drive structure
                os.path.join(self.base_dir, 'ASVspoof2019_LA_train', 'flac', key + '.flac'),  # /content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_train/flac/
                os.path.join(self.base_dir, 'ASVspoof2019_LA_dev', 'flac', key + '.flac'),    # /content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_dev/flac/
                os.path.join(self.base_dir, 'ASVspoof2019_LA_eval', 'flac', key + '.flac'),   # /content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_eval/flac/
                # Alternative structures
                os.path.join(self.base_dir, 'flac', key + '.flac'),        # /content/sample_data/data/flac/
                os.path.join(self.base_dir, 'LA', 'flac', key + '.flac'),  # /content/sample_data/data/LA/flac/
                os.path.join(self.base_dir, key + '.flac'),                # /content/sample_data/data/
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        X, _ = librosa.load(file_path, sr=16000)
                        y = self.labels[key]
                        return Tensor(pad(X, self.cut)), y
                    except (OSError, PermissionError) as e:
                        print(f"Warning: Cannot access {file_path}: {e}")
                        continue
            
            # If no path works, return zeros
            print(f"Warning: Could not find {key} in any expected path. Tried:")
            for path in file_paths:
                print(f"  - {path}")
            return torch.zeros(self.cut), 0
            
        except Exception as e:
            print(f"Error loading {key}: {e}. Returning zeros.")
            return torch.zeros(self.cut), 0

class Dataset_ASVspoof_eval(Dataset):
    def __init__(self, list_IDs, base_dir, cut=64600):
        self.list_IDs, self.base_dir, self.cut = list_IDs, base_dir, cut
    def __len__(self): return len(self.list_IDs)
    def __getitem__(self, index):
        key = self.list_IDs[index]
        try:
            # Local data structure paths for evaluation (after copying from Google Drive)
            file_paths = [
                # Primary local structure (after copy script)
                os.path.join(self.base_dir, 'ASVspoof2019_LA_eval', key + '.flac'),   # /content/sample_data/data/ASVspoof2019_LA_eval/
                os.path.join(self.base_dir, 'ASVspoof2019_LA_train', key + '.flac'),  # /content/sample_data/data/ASVspoof2019_LA_train/
                os.path.join(self.base_dir, 'ASVspoof2019_LA_dev', key + '.flac'),    # /content/sample_data/data/ASVspoof2019_LA_dev/
                # Fallback to original Google Drive structure
                os.path.join(self.base_dir, 'ASVspoof2019_LA_eval', 'flac', key + '.flac'),   # /content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_eval/flac/
                os.path.join(self.base_dir, 'ASVspoof2019_LA_train', 'flac', key + '.flac'),  # /content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_train/flac/
                os.path.join(self.base_dir, 'ASVspoof2019_LA_dev', 'flac', key + '.flac'),    # /content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_dev/flac/
                # Alternative structures
                os.path.join(self.base_dir, 'flac', key + '.flac'),        # /content/sample_data/data/flac/
                os.path.join(self.base_dir, 'LA', 'flac', key + '.flac'),  # /content/sample_data/data/LA/flac/
                os.path.join(self.base_dir, key + '.flac'),                # /content/sample_data/data/
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    try:
                        X, _ = librosa.load(file_path, sr=16000)
                        return Tensor(pad(X, self.cut)), key
                    except (OSError, PermissionError) as e:
                        print(f"Warning: Cannot access {file_path}: {e}")
                        continue
            
            # If no path works, return zeros
            print(f"Warning: Could not find {key} in any expected path. Tried:")
            for path in file_paths:
                print(f"  - {path}")
            return torch.zeros(self.cut), key
            
        except Exception as e:
            print(f"Error loading eval file {key}: {e}. Returning zeros.")
            return torch.zeros(self.cut), key

# ===================================================================
# Part 4: Main script logic
# ===================================================================
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_accuracy(dev_loader, model, device):
    num_correct, num_total = 0.0, 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x, batch_y = batch_x.to(device), batch_y.view(-1).type(torch.int64).to(device)
        with torch.no_grad():
            batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum().item()
    return 100 * (num_correct / num_total)

def produce_evaluation_file(dataset, model, device, save_path, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    with open(save_path, 'w') as fh:
        for batch_x, utt_id in data_loader:
            batch_x = batch_x.to(device)
            with torch.no_grad():
                batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            for f, cm in zip(utt_id, batch_score):
                fh.write(f'{f} {cm}\n')
    print(f'Scores saved to {save_path}')

def train_epoch(train_loader, model, optim, device, criterion, grad_clip_norm):
    running_loss, num_correct, num_total = 0, 0, 0
    model.train()
    if any(p.requires_grad for p in model.wav2vec2_extractor.model.parameters()):
        model.wav2vec2_extractor.model.train()

    for ii, (batch_x, batch_y) in enumerate(train_loader, 1):
        batch_x, batch_y = batch_x.to(device), batch_y.view(-1).type(torch.int64).to(device)
        
        optim.zero_grad()
        batch_out = model(batch_x)
        
        # Check for NaN/Inf in model output
        if torch.isnan(batch_out).any() or torch.isinf(batch_out).any():
            print(f"Warning: NaN/Inf detected in model output at batch {ii}, skipping...")
            continue
            
        batch_loss = criterion(batch_out, batch_y)
        
        # Check for NaN/Inf in loss
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            print(f"Warning: NaN/Inf detected in loss at batch {ii}, skipping...")
            continue
            
        batch_loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        
        optim.step()

        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum().item()
        running_loss += batch_loss.item() * batch_x.size(0)
        num_total += batch_x.size(0)
        
        if ii % 10 == 0:  # Print every 10 batches instead of every batch
            sys.stdout.write(f'\r\t Batch {ii}/{len(train_loader)}, Acc: {(num_correct/num_total)*100:.2f}%')
            sys.stdout.flush()
            
    return running_loss / num_total, num_correct / num_total * 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Maze6: Maze2 Foundation + Maze5 Features - ASVspoof2019 Compatible',
        epilog='''
Example usage:
  # Training with default local paths (after copying data):
  python maze6.py --database_path=/content/sample_data/data/
  
  # Training with Google Drive paths:
  python maze6.py --database_path=/content/drive/MyDrive/ASVspoof2019/Extract/LA/
  
  # Training with custom paths:
  python maze6.py --database_path=/path/to/ASVspoof2019/Extract/LA/ --protocols_path=/path/to/protocols/
  
  # Evaluation mode:
  python maze6.py --eval --model_path=models/Real_Maze_6/best_epoch_10_Real_Maze_6.pth
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Config argument no longer required - using hardcoded FMSL configuration
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/', 
                       help='Root path of ASVspoof2019 database (default: /content/sample_data/data/)')
    parser.add_argument('--protocols_path', type=str, default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/', 
                       help='Path to ASVspoof2019 protocol files (default: /content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/)')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)  # Optimal batch size for stability
    parser.add_argument('--lr', type=float, default=0.00005)  # Slower, more stable learning
    parser.add_argument('--lr_wav2vec2', type=float, default=0.000005)  # Even slower for Wav2Vec2
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # More regularization
    parser.add_argument('--loss', type=str, default='cce', choices=['focal', 'cce'])  # Use CCE for stability
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--comment', type=str, default="Maze6_Optimized", help='Comment for model directory')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'])
    parser.add_argument('--eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save evaluation scores')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='Minimum learning rate')
    
    args = parser.parse_args()
    
    # Learning rate optimization for better training
    if args.lr > 0.0001:
        print(f"⚠️  WARNING: Learning rate {args.lr} might be too high for stable training!")
        print(f"💡 Consider using a lower learning rate (e.g., 0.00005) for better convergence")
    elif args.lr < 0.00001:
        print(f"⚠️  WARNING: Learning rate {args.lr} might be too low for effective training!")
        print(f"💡 Consider using a higher learning rate (e.g., 0.00005) for faster convergence")
    
    # Display configuration
    print("🚀 Maze6: ASVspoof2019 Compatible Training Script")
    print("=" * 60)
    print(f"📁 Database path: {args.database_path}")
    print(f"📋 Protocols path: {args.protocols_path}")
    print(f"🎯 Track: {args.track}")
    print(f"💾 Model save path: models/{args.comment}")
    print(f"🎓 Learning rate: {args.lr}")
    print(f"⏰ Patience: {args.patience} epochs")
    print(f"📉 Min LR: {args.min_lr}")
    print("=" * 60)
    
    # Validate database paths before proceeding
    if not validate_database_paths(args.database_path, args.protocols_path):
        print("❌ Database validation failed. Please check your paths and try again.")
        sys.exit(1)
    
    # Improved model configuration for better training
    model_config_params = {
        'model': {
            'filts': [128, [128, 128], [128, 256]],  # Original architecture as required
            'first_conv': 251,
            'sample_rate': 16000,
            'nb_fc_node': 1024,
            'fc_dropout': 0.5,  # Standard dropout for regularization
            'nb_classes': 2,
            # Wav2Vec2 parameters - STANDARDIZED for fair comparison with FMSL version
            'wav2vec2_model_name': 'facebook/wav2vec2-large-960h',  # Use large model for fair comparison
            'wav2vec2_output_dim': 1024,  # Large model output dimension
            'wav2vec2_freeze_cnn': False,  # Allow Wav2Vec2 CNN fine-tuning
            'wav2vec2_unfrozen_transformers': 2,  # Unfreeze last 2 transformer layers
            'wav2vec2_output_layers': [0, 6, 12, 18, 24],  # Use multiple layers for richer features
            # Transformer parameters
            'transformer_num_layers': 4,  # Standard layers for better learning
            'transformer_nhead': 8,
            'transformer_dim_feedforward': 2048,  # Standard size for better capacity
            'transformer_dropout': 0.1,
            # SpecAugment for data augmentation
            'use_spec_augment_w2v2': True,
            'spec_aug_freq_mask_param_w2v2': 15,  # More aggressive augmentation
            'spec_aug_n_freq_masks_w2v2': 2,      # More masks
            'spec_aug_time_mask_param_w2v2': 15,  # More aggressive augmentation
            'spec_aug_n_time_masks_w2v2': 2,      # More masks
            # Attention pooling
            'use_attention_pooling': True,
            'attention_pooling_dim': 256
        }
    }
    print("Using STANDARDIZED configuration (1024 dim) for fair thesis comparison with FMSL version")
    
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    model_save_path = os.path.join('models', args.comment)
    if not os.path.exists(model_save_path): os.makedirs(model_save_path)
    
    model = Model_Maze6(model_config_params['model'], device)
    if args.model_path: model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable model parameters: {nb_params / 1e6:.2f} M')
    model = model.to(device)
    
    # Optimizer
    if model_config_params['model'].get('wav2vec2_unfrozen_transformers', 0) > 0 or \
       not model_config_params['model'].get('wav2vec2_freeze_cnn', True):
        w2v_params = [p for n, p in model.named_parameters() if 'wav2vec2' in n and p.requires_grad]
        other_params = [p for n, p in model.named_parameters() if 'wav2vec2' not in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': w2v_params, 'lr': args.lr_wav2vec2},
            {'params': other_params, 'lr': args.lr}], weight_decay=args.weight_decay)
        print(f"Optimizer: AdamW with differential LRs ({args.lr_wav2vec2} for w2v, {args.lr} for rest)")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"Optimizer: AdamW with single LR ({args.lr})")
        
    # Use balanced class weights for better training
    if args.loss == 'focal':
        criterion = FocalLoss(alpha=0.5, gamma=2.0).to(device)  # Balanced focal loss
    else:
        # Use balanced class weights for better training (less extreme than [0.1, 0.9])
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.3, 0.7]).to(device))
        print(f"Using balanced class weights [0.3, 0.7] for better training")
    print(f"Loss function: {args.loss}")
    
    # Add learning rate scheduler for better training
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.min_lr
    )
    print(f"Learning rate scheduler: CosineAnnealingLR (T_max={args.num_epochs}, min_lr={args.min_lr})")

    if args.eval:
        eval_protocol = os.path.join(args.protocols_path, f'ASVspoof2019.{args.track}.cm.eval.trl.txt')
        eval_files = genSpoof_list(dir_meta=eval_protocol, is_eval=True)
        eval_set = Dataset_ASVspoof_eval(list_IDs=eval_files, base_dir=args.database_path)
        produce_evaluation_file(eval_set, model, device, args.eval_output or os.path.join(model_save_path, 'scores.txt'), args.batch_size)
        sys.exit(0)
    
    train_protocol = os.path.join(args.protocols_path, f"ASVspoof2019.{args.track}.cm.train.trn.txt")
    d_label_trn, file_train = genSpoof_list(dir_meta=train_protocol, is_train=True)
    train_set = Dataset_ASVspoof_train(list_IDs=file_train, labels=d_label_trn, base_dir=args.database_path)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    
    dev_protocol = os.path.join(args.protocols_path, f"ASVspoof2019.{args.track}.cm.dev.trl.txt")
    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_protocol)
    dev_set = Dataset_ASVspoof_train(list_IDs=file_dev, labels=d_label_dev, base_dir=args.database_path)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    writer = SummaryWriter(f'logs/{args.comment}')
    best_acc, best_epoch = 0.0, 0
    patience_counter = 0

    for epoch in range(args.num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, device, criterion, args.grad_clip_norm)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        
        # Update learning rate scheduler (CosineAnnealingWarmRestarts doesn't need validation accuracy)
        scheduler.step()
        
        writer.add_scalar('accuracy/train', train_accuracy, epoch)
        writer.add_scalar('accuracy/validation', valid_accuracy, epoch)
        writer.add_scalar('loss/train', running_loss, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'\nEpoch {epoch + 1}/{args.num_epochs} | Loss: {running_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Valid Acc: {valid_accuracy:.2f}% | LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch+1}_{args.comment}.pth'))

        if valid_accuracy > best_acc:
            print(f'Best validation accuracy found: {valid_accuracy:.2f}% at epoch {epoch + 1}')
            best_acc, best_epoch = valid_accuracy, epoch + 1
            patience_counter = 0  # Reset patience counter
            
            for f in os.listdir(model_save_path):
                if f.startswith('best_epoch_'): os.remove(os.path.join(model_save_path, f))
            
            torch.save(model.state_dict(), os.path.join(model_save_path, f'best_epoch_{best_epoch}_{args.comment}.pth'))
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs (patience: {args.patience})')
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f'Early stopping triggered after {patience_counter} epochs without improvement')
            break
        
        # Stop if learning rate is too low
        if optimizer.param_groups[0]['lr'] < args.min_lr:
            print(f'Learning rate {optimizer.param_groups[0]["lr"]:.2e} below minimum {args.min_lr:.2e}, stopping training')
            break

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}% at epoch {best_epoch}.")
    writer.close()