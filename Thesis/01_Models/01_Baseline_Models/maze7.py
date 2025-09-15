#!/usr/bin/env python3
# ===================================================================
# maze7.py - Model 6: RawNet with Wav2Vec2 Features
# 
# DESCRIPTION:
# This model replaces SincConv with Wav2Vec2 pre-trained features for
# higher quality audio representations. This establishes the baseline
# for models using pre-trained audio encoders.
# 
# STANDARDIZATION:
# - Same data loading pipeline as all other maze files
# - Same training loop structure
# - Same evaluation metrics
# - Same hyperparameter ranges
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

# ===================================================================
# Part 1: Model Definition (Model 6 - RawNet + Wav2Vec2)
# ===================================================================

class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self, model_name='facebook/wav2vec2-base-960h', device='cuda', freeze_extractor=True):
        super(Wav2Vec2FeatureExtractor, self).__init__()
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = HuggingFaceWav2Vec2Model.from_pretrained(model_name).to(device)
        
        if freeze_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.eval()

    def forward(self, x_waveforms_list):
        if isinstance(x_waveforms_list, torch.Tensor) and x_waveforms_list.ndim == 3:
            x_waveforms_list = x_waveforms_list.squeeze(1)
        
        # Convert to numpy for processor
        if isinstance(x_waveforms_list, torch.Tensor):
            x_waveforms_list = x_waveforms_list.cpu().numpy()

        # Debug: Check for invalid values
        if np.any(np.isnan(x_waveforms_list)) or np.any(np.isinf(x_waveforms_list)):
            print(f"WARNING: Invalid values in input: NaN={np.any(np.isnan(x_waveforms_list))}, Inf={np.any(np.isinf(x_waveforms_list))}")
            # Replace invalid values with zeros
            x_waveforms_list = np.nan_to_num(x_waveforms_list, nan=0.0, posinf=0.0, neginf=0.0)

        inputs = self.processor(x_waveforms_list, return_tensors='pt', padding=True, sampling_rate=16000)
        
        with torch.set_grad_enabled(not self.model.training):
            outputs = self.model(
                input_values=inputs.input_values.to(self.device),
                attention_mask=getattr(inputs, 'attention_mask', None).to(self.device) if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None else None
            )
        
        # Extract features and permute to match expected format
        features = outputs.last_hidden_state
        
        # Debug: Check for invalid features
        if torch.any(torch.isnan(features)) or torch.any(torch.isinf(features)):
            print(f"WARNING: Invalid values in Wav2Vec2 features: NaN={torch.any(torch.isnan(features))}, Inf={torch.any(torch.isinf(features))}")
            # Replace invalid values with zeros
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
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
        
        self.activation = nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1)
        return x * y.expand_as(x)

class Model6_RawNet_Wav2Vec2(nn.Module):
    def __init__(self, d_args, device):
        super(Model6_RawNet_Wav2Vec2, self).__init__()
        self.device = device

        # Wav2Vec2 feature extractor
        self.wav2vec2_extractor = Wav2Vec2FeatureExtractor(
            model_name=d_args.get('wav2vec2_model_name', 'facebook/wav2vec2-base-960h'),
            device=self.device,
            freeze_extractor=d_args.get('wav2vec2_freeze', True)
        )
        
        # Get Wav2Vec2 output dimension
        wav2vec2_out_dim = d_args['wav2vec2_output_dim']
        
        # Feature projection if needed
        self.feature_projection = nn.Conv1d(wav2vec2_out_dim, d_args['filts'][0], kernel_size=1) if wav2vec2_out_dim != d_args['filts'][0] else nn.Identity()
        
        self.first_bn = nn.BatchNorm1d(d_args['filts'][0])
        self.selu = nn.SELU(inplace=False)

        # Residual blocks with SE
        self.block0 = Residual_Block_SE([d_args['filts'][0], d_args['filts'][0]], first=True, stride=1)
        self.se0 = SEBlock(d_args['filts'][0])
        
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
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(d_args['filts'][2][1], d_args['nb_fc_node'])  # Use last layer's output dimension
        self.dropout_fc = nn.Dropout(p=d_args.get('fc_dropout', 0.5))
        self.fc2 = nn.Linear(d_args['nb_fc_node'], d_args['nb_classes'])
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(1)
            
        # Wav2Vec2 feature extraction
        out = self.wav2vec2_extractor(x)
        
        # Feature projection if needed
        out = self.feature_projection(out)
        out = self.selu(self.first_bn(out))
        
        # Residual blocks with SE
        out = self.se0(self.block0(out))
        for block, se in zip(self.res_blocks, self.se_blocks):
            out = se(block(out))
        
        # Global pooling
        out = self.avg_pool(out).squeeze(-1)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        return self.logsoftmax(out)

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
                os.path.join(self.base_dir, 'LA', 'flac', key + '.flac'),  # /content/sample_data/data/LA/flac/
                os.path.join(self.base_dir, 'flac', key + '.flac'),        # /content/sample_data/data/flac/
                os.path.join(self.base_dir, key + '.flac')                 # /content/sample_data/data/
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    X, _ = librosa.load(file_path, sr=16000)
                    y = self.labels[key]
                    return Tensor(pad(X, self.cut)), y
            
            # If no path works, return zeros
            print(f"Warning: Could not find {key} in any expected path")
            return torch.zeros(self.cut), 0
            
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
                os.path.join(self.base_dir, 'LA', 'flac', key + '.flac'),  # /content/sample_data/data/LA/flac/
                os.path.join(self.base_dir, 'flac', key + '.flac'),        # /content/sample_data/data/flac/
                os.path.join(self.base_dir, key + '.flac')                 # /content/sample_data/data/
            ]
            
            for file_path in file_paths:
                if os.path.exists(file_path):
                    X, _ = librosa.load(file_path, sr=16000)
                    return Tensor(pad(X, self.cut)), key
            
            # If no path works, return zeros
            print(f"Warning: Could not find {key} in any expected path")
            return torch.zeros(self.cut), key
            
        except Exception as e:
            print(f"Error loading {key}: {e}. Returning zeros.")
            return torch.zeros(self.cut), key

# ===================================================================
# Part 3: Training Functions (Standardized)
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
        batch_out = model(batch_x)
        
        # Debug: Check for invalid outputs
        if torch.any(torch.isnan(batch_out)) or torch.any(torch.isinf(batch_out)):
            print(f"WARNING: Invalid model outputs detected in batch {ii}: NaN={torch.any(torch.isnan(batch_out))}, Inf={torch.any(torch.isinf(batch_out))}")
            # Skip this batch
            continue
        
        batch_loss = criterion(batch_out, batch_y)
        
        # Debug: Check for invalid loss
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            print(f"WARNING: Invalid loss detected in batch {ii}: {batch_loss.item()}")
            # Skip this batch
            continue
        
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
            batch_out = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
    
    return 100 * (num_correct / num_total)

def produce_evaluation_file(dataset, model, device, save_path, batch_size=128):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    
    with open(save_path, 'w') as fh:
        with torch.no_grad():
            for batch_x, utt_id in data_loader:
                batch_size = batch_x.size(0)
                batch_x = batch_x.to(device)
                batch_out = model(batch_x)
                batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
                
                for f, cm in zip(utt_id, batch_score):
                    fh.write(f'{f} {cm}\n')
    
    print(f'Scores saved to {save_path}')

# ===================================================================
# Part 4: Main Training Script
# ===================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Maze7: RawNet with Wav2Vec2 Features')
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/', 
                       help='Root path of database (use local path for efficiency)')
    parser.add_argument('--protocols_path', type=str, 
                       default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/', 
                       help='Path to protocol files')
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='cce', choices=['cce', 'focal'])
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None, help='Path to load a pre-trained model')
    parser.add_argument('--comment', type=str, default="Real_Maze_7", help='Comment for model directory')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'])
    parser.add_argument('--eval', action='store_true', help='Enable evaluation mode')
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save evaluation scores')
    
    args = parser.parse_args()
    
    # Standardized model configuration for fair comparison with FMSL
    model_config = {
        'model': {
            'filts': [128, [128, 128], [128, 256]],  # FMSL-compatible architecture
            'nb_fc_node': 1024,
            'fc_dropout': 0.5,
            'nb_classes': 2,
            # Wav2Vec2 parameters
            'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
            'wav2vec2_output_dim': 768,  # Base model output dimension
            'wav2vec2_freeze': True  # Freeze Wav2Vec2 for efficiency
        }
    }
    
    set_random_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Create model save directory
    model_save_path = os.path.join('models', args.comment)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    # Initialize model
    model = Model6_RawNet_Wav2Vec2(model_config['model'], device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable model parameters: {nb_params / 1e6:.2f} M')
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW with LR {args.lr}")
    
    # Loss function
    if args.loss == 'focal':
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    print(f"Loss function: {args.loss}")

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
