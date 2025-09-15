# ===================================================================
# main_combined.py (Version 3.1 - Corrected for Model 7)
# This script combines the new Model7, data_utils.py, and main.py.
# Fixes:
# - Corrected AttributeError for 'attention_mask'.
# ===================================================================


# ===================================================================
# Part 1: Combined Imports from All Files
# ===================================================================
import argparse
import sys
import os
import yaml
import librosa
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

# New imports required by Model 7
from transformers import Wav2Vec2Processor, Wav2Vec2Model as HuggingFaceWav2Vec2Model
import torchaudio.transforms as T


# ===================================================================
# Part 2: Contents of the NEW model.py (Model 7)
# ===================================================================

class Residual_Block_SE(nn.Module):
    def __init__(self, nb_filts_in_out, first=False, dropout_rate=0.3, stride=1):
        super(Residual_Block_SE, self).__init__()
        self.first, self.nb_filts_in, self.nb_filts_out = first, nb_filts_in_out[0], nb_filts_in_out[1]
        self.dropout_rate, self.stride = dropout_rate, stride
        if not self.first:
            self.bn1 = nn.BatchNorm1d(self.nb_filts_in)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.conv1 = nn.Conv1d(self.nb_filts_in, self.nb_filts_out, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(self.nb_filts_out)
        self.conv2 = nn.Conv1d(self.nb_filts_out, self.nb_filts_out, kernel_size=3, padding=1, stride=1)
        self.downsample = (self.nb_filts_in != self.nb_filts_out) or (self.stride != 1 and self.first)
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

class Wav2Vec2FeatureExtractor(nn.Module):
    def __init__(self, model_name='facebook/wav2vec2-large-960h', device='cuda', freeze_extractor=True):
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
        
        # The processor expects a list of 1D numpy arrays or a 2D tensor
        if isinstance(x_waveforms_list, torch.Tensor):
             x_waveforms_list = x_waveforms_list.cpu().numpy()

        inputs = self.processor(x_waveforms_list, return_tensors='pt', padding=True, sampling_rate=16000)
        
        # === BUG FIX IS HERE ===
        # Safely get the attention mask. It will be None if the key doesn't exist.
        attention_mask = inputs.get('attention_mask')
        
        with torch.set_grad_enabled(not self.model.training):
            outputs = self.model(
                input_values=inputs.input_values.to(self.device),
                # Use the safely retrieved attention mask
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None
            )
        features = outputs.last_hidden_state.permute(0, 2, 1)
        return features

class Model7_RawNet_Wav2Vec2_SpecAug_FocalLoss(nn.Module):
    def __init__(self, d_args, device):
        super(Model7_RawNet_Wav2Vec2_SpecAug_FocalLoss, self).__init__()
        self.device = device

        self.wav2vec2_extractor = Wav2Vec2FeatureExtractor(
            model_name=d_args.get('wav2vec2_model_name', 'facebook/wav2vec2-large-960h'),
            device=self.device,
            freeze_extractor=d_args.get('wav2vec2_freeze', True)
        )
        wav2vec2_out_dim = d_args['wav2vec2_output_dim']

        self.first_bn = nn.BatchNorm1d(wav2vec2_out_dim)
        self.selu = nn.SELU(inplace=False)

        self.spec_augment = nn.Sequential()
        if d_args.get('use_spec_augment_w2v2', False):
            self.spec_augment.add_module("freq_mask_w2v2", T.FrequencyMasking(freq_mask_param=d_args['spec_aug_freq_mask_param_w2v2']))
            for i in range(d_args.get('spec_aug_n_freq_masks_w2v2', 1) - 1):
                self.spec_augment.add_module(f"freq_mask_w2v2_{i+1}", T.FrequencyMasking(freq_mask_param=d_args['spec_aug_freq_mask_param_w2v2']))
            self.spec_augment.add_module("time_mask_w2v2", T.TimeMasking(time_mask_param=d_args['spec_aug_time_mask_param_w2v2']))
            for i in range(d_args.get('spec_aug_n_time_masks_w2v2', 1) - 1):
                self.spec_augment.add_module(f"time_mask_w2v2_{i+1}", T.TimeMasking(time_mask_param=d_args['spec_aug_time_mask_param_w2v2']))

        self.block0 = Residual_Block_SE([wav2vec2_out_dim, d_args['filts'][0]], first=True, dropout_rate=d_args.get('dropout_rate', 0.3), stride=d_args.get('res_pool_stride_w2v2', 1))
        self.se0 = SEBlock(d_args['filts'][0])
        self.block1 = Residual_Block_SE([d_args['filts'][0], d_args['filts'][1][0]], dropout_rate=d_args.get('dropout_rate', 0.3), stride=d_args.get('res_pool_stride_w2v2', 2))
        self.se1 = SEBlock(d_args['filts'][1][0])
        self.block2 = Residual_Block_SE([d_args['filts'][1][0], d_args['filts'][1][1]], dropout_rate=d_args.get('dropout_rate', 0.3), stride=d_args.get('res_pool_stride_w2v2', 2))
        self.se2 = SEBlock(d_args['filts'][1][1])
        self.block3 = Residual_Block_SE([d_args['filts'][1][1], d_args['filts'][2][0]], dropout_rate=d_args.get('dropout_rate', 0.3), stride=d_args.get('res_pool_stride_w2v2', 2))
        self.se3 = SEBlock(d_args['filts'][2][0])
        self.block4 = Residual_Block_SE([d_args['filts'][2][0], d_args['filts'][2][1]], dropout_rate=d_args.get('dropout_rate', 0.3), stride=d_args.get('res_pool_stride_w2v2', 2))
        self.se4 = SEBlock(d_args['filts'][2][1])
        self.block5 = Residual_Block_SE([d_args['filts'][2][1], d_args['filts'][2][1]], dropout_rate=d_args.get('dropout_rate', 0.3), stride=d_args.get('res_pool_stride_w2v2', 2))
        self.se5 = SEBlock(d_args['filts'][2][1])

        transformer_dim = d_args['filts'][2][1]  # Use the last layer's output dimension
        self.bn_before_transformer = nn.BatchNorm1d(transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=d_args.get('transformer_nhead', 8),
            dim_feedforward=d_args.get('transformer_dim_feedforward', 2048),
            dropout=d_args.get('transformer_dropout', 0.1), activation='relu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=d_args.get('transformer_num_layers', 6))

        self.avgpool_after_blocks = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(transformer_dim, d_args['nb_fc_node'])
        self.dropout_fc = nn.Dropout(p=d_args.get('fc_dropout', 0.5))
        self.fc2 = nn.Linear(d_args['nb_fc_node'], d_args['nb_classes'])
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x_wave):
        if x_wave.ndim == 3 and x_wave.size(1) == 1:
            x_wave = x_wave.squeeze(1)
        out = self.wav2vec2_extractor(x_wave)
        out = self.first_bn(out)
        out = self.selu(out)
        if self.training and hasattr(self, 'spec_augment') and len(list(self.spec_augment.children())) > 0:
            out = self.spec_augment(out)
        out = self.se0(self.block0(out))
        out = self.se1(self.block1(out))
        out = self.se2(self.block2(out))
        out = self.se3(self.block3(out))
        out = self.se4(self.block4(out))
        out = self.se5(self.block5(out))
        out = self.bn_before_transformer(out)
        out = out.permute(0, 2, 1) # Permute for Transformer
        out = self.transformer_encoder(out)
        out = out.permute(0, 2, 1) # Permute back
        out = self.avgpool_after_blocks(out).squeeze(-1)
        out = self.fc1(out)
        out = self.dropout_fc(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss


# ===================================================================
# Part 3: Contents of data_utils.py (Unchanged)
# ===================================================================
def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()
    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list
    elif is_eval:
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else: # for dev set
        for line in l_meta:
            _, key, _, _, label = line.strip().split(' ')
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # Try multiple path structures for compatibility
        file_paths = [
            os.path.join(self.base_dir, 'LA', 'flac', key + '.flac'),  # /content/sample_data/data/LA/flac/
            os.path.join(self.base_dir, 'flac', key + '.flac'),        # /content/sample_data/data/flac/
            os.path.join(self.base_dir, key + '.flac')                 # /content/sample_data/data/
        ]
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                X, fs = librosa.load(file_path, sr=16000)
                X_pad = pad(X, self.cut)
                x_inp = Tensor(X_pad)
                y = self.labels[key]
                return x_inp, y
        
        # If no path works, return zeros
        print(f"Warning: Could not find {key} in any expected path")
        return torch.zeros(self.cut), 0

class Dataset_ASVspoof2021_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # Try multiple path structures for compatibility
        file_paths = [
            os.path.join(self.base_dir, 'LA', 'flac', key + '.flac'),  # /content/sample_data/data/LA/flac/
            os.path.join(self.base_dir, 'flac', key + '.flac'),        # /content/sample_data/data/flac/
            os.path.join(self.base_dir, key + '.flac')                 # /content/sample_data/data/
        ]
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                X, fs = librosa.load(file_path, sr=16000)
                X_pad = pad(X, self.cut)
                x_inp = Tensor(X_pad)
                return x_inp, key
        
        # If no path works, return zeros
        print(f"Warning: Could not find {key} in any expected path")
        return torch.zeros(self.cut), key


# ===================================================================
# Part 4: Main script logic from main.py (Modified for new model)
# ===================================================================
def set_random_seed(seed, args):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if args.cudnn_deterministic_toggle:
        torch.backends.cudnn.deterministic = True
    if args.cudnn_benchmark_toggle:
        torch.backends.cudnn.benchmark = True

def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False, num_workers=0)
    model.eval()
    with open(save_path, 'w') as fh:
        for batch_x, utt_id in data_loader:
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            for f, cm in zip(utt_id, batch_score):
                fh.write(f'{f} {cm}\n')
    print(f'Scores saved to {save_path}')

def train_epoch(train_loader, model, optim, device, loss_fn):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    model.train()

    for ii, (batch_x, batch_y) in enumerate(train_loader, 1):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        
        batch_loss = loss_fn(batch_out, batch_y)
        
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        
        if ii % 1 == 0:
            sys.stdout.write(f'\r\t Processing batch {ii}/{len(train_loader)}, Current Acc: {(num_correct / num_total) * 100:.2f}%')
            sys.stdout.flush()
            
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        
    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system with Model 7')
    
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/', help='Root path of database (use local path for efficiency)')
    parser.add_argument('--protocols_path', type=str, default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/')
    
    parser.add_argument('--batch_size', type=int, default=16) # Reduced batch size for large model
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='focal_loss', help='loss function, e.g., focal_loss or CCE')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--model_path', type=str, default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default="Real_Maze_2", help='Comment for model tag')
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA', 'DF'])
    parser.add_argument('--eval_output', type=str, default=None, help='Path to save evaluation result')
    parser.add_argument('--eval', action='store_true', default=False, help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False, help='eval database')
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', default=True)
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', default=False)
    
    # Standardized model configuration for fair comparison with FMSL
    model_config = {
        'model': {
            'filts': [128, [128, 128], [128, 256]],  # FMSL-compatible architecture
            'first_conv': 251,
            'sample_rate': 16000,
            'nb_fc_node': 1024,
            'fc_dropout': 0.5,
            'nb_classes': 2,
            # Wav2Vec2 parameters
            'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
            'wav2vec2_output_dim': 768,  # Base model output dimension
            'wav2vec2_freeze': True,  # Freeze Wav2Vec2 for efficiency
            # SpecAugment parameters
            'use_spec_augment_raw': True,
            'spec_aug_freq_mask_param_raw': 10,
            'spec_aug_n_freq_masks_raw': 1,
            'spec_aug_time_mask_param_raw': 10,
            'spec_aug_n_time_masks_raw': 1
        }
    }
    print("Using hardcoded FMSL-compatible configuration")
    
    args = parser.parse_args()
    
    if not os.path.exists('models'):
        os.mkdir('models')

    set_random_seed(args.seed, args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    model = Model7_RawNet_Wav2Vec2_SpecAug_FocalLoss(model_config['model'], device)
    
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print(f'Number of model parameters: {nb_params / 1e6:.2f} M')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Model loaded from {args.model_path}')

    if args.loss == 'focal_loss':
        criterion = FocalLoss().to(device)
        print("Using FocalLoss")
    else:
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)
        print("Using weighted CrossEntropyLoss")

    if args.eval:
        eval_protocol_file = f'ASVspoof2019.LA.cm.eval.trl.txt'
        eval_protocol_path = os.path.join(args.protocols_path, eval_protocol_file)
        file_eval = genSpoof_list(dir_meta=eval_protocol_path, is_eval=True)
        print(f'Number of eval trials: {len(file_eval)}')
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=file_eval, base_dir=os.path.join(args.database_path, f'ASVspoof2019_{args.track}_eval/'))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)
    
    model_tag = f'model_{args.track}_{args.loss}_{args.num_epochs}_{args.batch_size}_{args.lr}'
    if args.comment:
        model_tag += f'_{args.comment}'
    model_save_path = os.path.join('models', model_tag)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    writer = SummaryWriter(f'logs/{model_tag}')
    
    train_protocol_path = os.path.join(args.protocols_path, f"ASVspoof2019.LA.cm.train.trn.txt")
    d_label_trn, file_train = genSpoof_list(dir_meta=train_protocol_path, is_train=True)
    print(f'Number of training trials: {len(file_train)}')
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train, labels=d_label_trn, base_dir=os.path.join(args.database_path, f'ASVspoof2019_{args.track}_train/'))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False)
    del train_set, d_label_trn
    
    dev_protocol_path = os.path.join(args.protocols_path, f"ASVspoof2019.LA.cm.dev.trl.txt")
    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_protocol_path)
    print(f'Number of validation trials: {len(file_dev)}')
    dev_set = Dataset_ASVspoof2019_train(list_IDs=file_dev, labels=d_label_dev, base_dir=os.path.join(args.database_path, f'ASVspoof2019_{args.track}_dev/'))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    del dev_set, d_label_dev

    best_acc = 0.0
    for epoch in range(args.num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader, model, optimizer, device, criterion)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print(f'\nEpoch {epoch + 1}/{args.num_epochs} | Loss: {running_loss:.4f} | Train Acc: {train_accuracy:.2f}% | Valid Acc: {valid_accuracy:.2f}%')
        
        if valid_accuracy > best_acc:
            print(f'Best validation accuracy found: {valid_accuracy:.2f}% at epoch {epoch + 1}')
            best_acc = valid_accuracy
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best_model.pth'))
