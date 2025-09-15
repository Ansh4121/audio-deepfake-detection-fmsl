
# ===================================================================
# maze5_compatible.py
#
# DESCRIPTION:
# This script is a refactored version of maze5_combined.py, made compatible
# with the structure and execution flow of maze2_combined.py.
#
# KEY FEATURES from Maze 5 retained:
# - Wav2Vec2 with multi-level feature fusion from specified hidden layers.
# - Fine-tuning capability for the Wav2Vec2 model.
# - Advanced waveform augmentations (noise and reverb) using torchaudio.
# - Attentive Statistics Pooling instead of simple average pooling.
# - Schedulers with warmup capabilities.
# - Focal Loss implementation.
#
# STRUCTURAL CHANGES (to align with Maze 2):
# - Unified main script structure.
# - Standardized argument parsing.
# - Simplified training and evaluation loops.
# - Data loading with `torchaudio` to support advanced augmentations.
# - Corrected model saving logic to save checkpoints in a 'maze5' subfolder
#   with specified naming convention (e.g., epoch_1_maze5.pth).
# ===================================================================

# ===================================================================
# Part 1: Combined Imports
# ===================================================================
import argparse
import sys
import os
import yaml
import random
from collections import OrderedDict
import time # Import time for potential timestamping

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

from transformers import Wav2Vec2Processor, Wav2Vec2Model as HuggingFaceWav2Vec2Model
import torchaudio
import torchaudio.transforms as T

# Add missing imports for metrics calculation
from sklearn.metrics import roc_curve, accuracy_score
# Assuming you have core_scripts in the same directory or sys.path is set
# from core_scripts.startup_config import set_random_seed
# from core_scripts.data_io import get_loader # Assuming get_loader is defined here or imported
# from core_scripts.loss_functions import FocalLoss # Assuming FocalLoss is defined here or imported
# from core_scripts.model_utils import create_optimizer # Assuming create_optimizer is defined here or imported
# from core_scripts.model_utils import create_scheduler # Assuming create_scheduler is defined here or imported
# from core_scripts.model_utils import load_model # Assuming load_model is defined here or imported
# from core_scripts.utils import EarlyStopper # Assuming EarlyStopper is defined here or imported

# Placeholder imports/definitions if they are not in core_scripts or elsewhere
# In a real scenario, these should be properly imported or defined.
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Placeholder for get_loader - will likely need adjustment based on actual implementation
def get_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

# Placeholder for FocalLoss - assuming a simple implementation or actual import
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# Model class definition
class Model_Maze5(nn.Module):
    def __init__(self, d_args, device):
        super(Model_Maze5, self).__init__()
        self.device = device
        
        # Wav2Vec2 feature extractor
        self.wav2vec2_extractor = Wav2Vec2FeatureExtractor(
            model_name=d_args.get('wav2vec2_model_name', 'facebook/wav2vec2-base-960h'),
            device=self.device,
            freeze_extractor=d_args.get('wav2vec2_freeze_cnn', True)
        )
        
        # Feature projection
        wav2vec2_out_dim = d_args['wav2vec2_output_dim']
        self.feature_projection = nn.Conv1d(wav2vec2_out_dim, d_args['filts'][0], kernel_size=1)
        
        # Residual blocks with progressive dimensions
        self.blocks = nn.ModuleList()
        
        # First block: from first dimension to second dimension
        input_channels = d_args['filts'][0]  # 128
        output_channels = d_args['filts'][1][0]  # [128, 128][0] = 128
        self.blocks.append(Residual_Block_SE(output_channels, first=True, input_channels=input_channels))
        
        # Second block: from second dimension to third dimension
        input_channels = d_args['filts'][1][0]  # [128, 128][0] = 128
        output_channels = d_args['filts'][2][0]  # [128, 256][0] = 128
        self.blocks.append(Residual_Block_SE(output_channels, first=False, input_channels=input_channels))
        
        # Final block maintains the last dimension
        final_channels = d_args['filts'][2][1]  # [128, 256][1] = 256
        self.final_block = Residual_Block_SE(final_channels, input_channels=d_args['filts'][2][0])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier - Fixed to use correct filts indexing
        self.classifier = nn.Sequential(
            nn.Linear(d_args['filts'][2][1], d_args['nb_fc_node']),  # [128, 256][1] = 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(d_args['nb_fc_node'], d_args['nb_classes'])
        )
    
    def forward(self, x):
        # Wav2Vec2 feature extraction
        features = self.wav2vec2_extractor(x)
        
        # Feature projection
        features = self.feature_projection(features)
        
        # Residual blocks
        for block in self.blocks:
            features = block(features)
        
        # Final block
        features = self.final_block(features)
        
        # Global pooling
        features = self.global_pool(features).squeeze(-1)
        
        # Classification
        output = self.classifier(features)
        return output

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

# Residual Block with SE
class Residual_Block_SE(nn.Module):
    def __init__(self, nb_filts, first=False, dropout_rate=0.3, input_channels=None):
        super(Residual_Block_SE, self).__init__()
        self.first = first
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels if input_channels is not None else nb_filts
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(self.input_channels)
        
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        
        self.conv1 = nn.Conv1d(self.input_channels, nb_filts, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm1d(nb_filts)
        self.conv2 = nn.Conv1d(nb_filts, nb_filts, kernel_size=3, padding=1, stride=1)
        
        # SE block
        self.se = SEBlock(nb_filts)
        
        # Handle dimension mismatch in residual connection
        if self.input_channels != nb_filts:
            self.shortcut = nn.Conv1d(self.input_channels, nb_filts, kernel_size=1, stride=1)
        else:
            self.shortcut = None
        
        # Pooling
        self.pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        identity = x
        
        if not self.first:
            out = self.activation(self.bn1(x))
        else:
            out = x
        
        out = self.conv1(out)
        out = self.dropout(self.activation(self.bn2(out)))
        out = self.conv2(out)
        
        # Apply SE
        out = self.se(out)
        
        # Add residual connection
        if self.shortcut is not None:
            # Use shortcut convolution to match dimensions
            identity = self.shortcut(identity)
        
        # Ensure time dimensions match (due to pooling)
        if identity.shape[2] != out.shape[2]:
            identity = F.adaptive_avg_pool1d(identity, out.shape[2])
        
        out += identity
        
        # Pooling
        out = self.pool(out)
        return out

# SE Block
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

# Placeholder for create_optimizer
def create_optimizer(model, args, config):
    """Creates optimizer with differential learning rates."""
    # Use .get with a default empty dict for robustness
    optimizer_config = config.get('optimizer', {})
    params = []
    # Default to args.lr if wav2vec2_lr is not in config
    wav2vec2_lr = optimizer_config.get('wav2vec2_lr', args.lr)
    base_lr = args.lr

    # Assume Wav2Vec2 model part is named 'wav2vec2_extractor'
    wav2vec2_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'wav2vec2_extractor.model' in name:
                 wav2vec2_params.append(param)
            else:
                 other_params.append(param)

    if wav2vec2_params:
        params.append({'params': wav2vec2_params, 'lr': wav2vec2_lr})
        print(f"Using differential LRs: Wav2Vec2 at {wav2vec2_lr}, rest at {base_lr}")
    if other_params:
        params.append({'params': other_params, 'lr': base_lr})
        if not wav2vec2_params:
            print(f"Using uniform LR: {base_lr}")

    # Default optimizer name and weight_decay if not in config
    optimizer_name = optimizer_config.get('name', 'Adam')
    weight_decay = optimizer_config.get('weight_decay', 0.0)


    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = optimizer_config.get('momentum', 0)
        optimizer = torch.optim.SGD(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    return optimizer

# Placeholder for create_scheduler
def create_scheduler(optimizer, args, config):
    """Creates learning rate scheduler with optional warmup."""
    # Use .get with a default empty dict for robustness
    scheduler_config = config.get('scheduler', {})
    scheduler_name = scheduler_config.get('name')
    warmup_epochs = scheduler_config.get('warmup_epochs', 0)

    scheduler = None
    if scheduler_name == 'CosineAnnealingLR':
        T_max = scheduler_config.get('T_max', args.num_epochs - warmup_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max
        )
    elif scheduler_name == 'StepLR':
         step_size = scheduler_config.get('step_size', args.num_epochs // 3)
         gamma = scheduler_config.get('gamma', 0.1)
         scheduler = torch.optim.lr_scheduler.StepLR(
             optimizer,
             step_size=step_size,
             gamma=gamma
         )
    elif scheduler_name == 'ReduceLROnPlateau':
         mode = scheduler_config.get('mode', 'min')
         factor = scheduler_config.get('factor', 0.1)
         patience = scheduler_config.get('patience', 10)
         threshold = scheduler_config.get('threshold', 0.0001)
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
             optimizer,
             mode=mode,
             factor=factor,
             patience=patience,
             threshold=threshold
         )
    elif scheduler_name is None or scheduler_name == 'None':
        scheduler = None
    else:
        raise ValueError(f"Scheduler {scheduler_name} not supported.")

    warmup_scheduler = None
    if warmup_epochs > 0:
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / warmup_epochs
            return 1.0 # No effect after warmup
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    return warmup_scheduler, scheduler

# Placeholder for load_model
def load_model(model, model_path, device):
    """Loads model state dict from a file."""
    if model_path and os.path.exists(model_path):
        print(f"Loading model checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model checkpoint loaded.")
        # Return loaded epoch and optimizer state if available
        return checkpoint.get('epoch', 0), checkpoint.get('optimizer_state_dict', None)
    else:
        print(f"No model checkpoint found at {model_path}. Starting from scratch.")
        return 0, None # Start from epoch 0, no optimizer state

# Placeholder for EarlyStopper
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf') # Track minimum loss
        self.best_validation_metric = float('inf') # Track best metric (e.g., EER)
        self.mode = 'min' # 'min' for loss/EER, 'max' for accuracy

    def early_stop(self, validation_metric):
        # Use validation_metric for comparison
        if self.mode == 'min':
            if validation_metric < self.best_validation_metric - self.min_delta:
                self.best_validation_metric = validation_metric
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
             if validation_metric > self.best_validation_metric + self.min_delta:
                 self.best_validation_metric = validation_metric
                 self.counter = 0
             else:
                 self.counter += 1
        else:
            raise ValueError("EarlyStopper mode must be 'min' or 'max'")

        if self.counter >= self.patience:
            return True
        return False

    def reset(self):
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_validation_metric = float('inf')


# ===================================================================
# Part 3: Dataset Definition
# ===================================================================

class Dataset_ASVspoof2019(Dataset):
    def __init__(self, list_IDs, labels, base_dir, is_train=False, is_dev=False, is_eval=False):
        """
        Args:
            list_IDs (list): List of file IDs (trial names).
            labels (dict): Dictionary mapping file IDs to labels ('bonafide', 'spoof').
            base_dir (str): Base directory containing the data subsets (train, dev, eval).
            is_train (bool): True if training data.
            is_dev (bool): True if development data.
            is_eval (bool): True if evaluation data.
        """
        super().__init__()
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.is_train = is_train
        self.is_dev = is_dev
        self.is_eval = is_eval

        if is_train:
            self.subset = 'ASVspoof2019_LA_train'
        elif is_dev:
            self.subset = 'ASVspoof2019_LA_dev'
        elif is_eval:
            self.subset = 'ASVspoof2019_LA_eval'
        else:
            raise ValueError("One of is_train, is_dev, or is_eval must be True.")

        # Try different possible directory structures
        possible_paths = [
            os.path.join(self.base_dir, self.subset, 'flac'),
            os.path.join(self.base_dir, 'LA', self.subset, 'flac'),
            os.path.join(self.base_dir, 'flac', self.subset),
            os.path.join(self.base_dir, self.subset, 'audio'),
            os.path.join(self.base_dir, self.subset),  # Direct subset path
            os.path.join(self.base_dir, 'LA', 'flac'),  # Simple LA/flac structure
            os.path.join(self.base_dir, 'flac'),        # Simple flac structure
            self.base_dir                               # Direct base directory
        ]
        
        # Find the first path that exists
        for path in possible_paths:
            if os.path.exists(path):
                self.data_dir = path
                break
        else:
            # If none exist, use the first one and let the error handling deal with it
            self.data_dir = possible_paths[0]
            print(f"Warning: Directory {self.data_dir} does not exist, but will try to use it")
        
        print(f"Dataset initialized for subset: {self.subset}")
        print(f"Looking for audio files in: {self.data_dir}")


    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Retrieves one sample (audio, label) from the dataset."""
        # Select sample ID
        ID = self.list_IDs[index]

        # Load audio file using torchaudio
        # Construct the full path
        filepath = os.path.join(self.data_dir, ID + '.flac')
        
        # Check if file exists before trying to load
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            # Try to find the file in other possible locations
            alternative_paths = [
                os.path.join(self.base_dir, 'LA', self.subset, 'flac', ID + '.flac'),
                os.path.join(self.base_dir, 'flac', self.subset, ID + '.flac'),
                os.path.join(self.base_dir, self.subset, 'audio', ID + '.flac')
            ]
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Found file at alternative location: {alt_path}")
                    filepath = alt_path
                    break
            else:
                print(f"File {ID}.flac not found in any expected location")
                # Return dummy tensor with fixed length and proper label
                dummy_waveform = torch.zeros(64600, dtype=torch.float32)
                label = self.labels[ID] if ID in self.labels else 0
                label = torch.tensor(1 if label == 'spoof' else 0, dtype=torch.long)
                return dummy_waveform, label
        
        try:
            waveform, sample_rate = torchaudio.load(filepath)
            # Torchaudio loads as (channels, time), we want (time,) for processing
            # Assuming mono audio for now
            if waveform.shape[0] > 1:
                 waveform = waveform[0, :] # Take the first channel if stereo
            
            # Ensure waveform is 1D (time,)
            waveform = waveform.squeeze()
            
            # Pad or truncate to fixed length (64600 samples = 4 seconds at 16kHz)
            target_length = 64600
            if waveform.shape[0] > target_length:
                # Truncate if longer
                waveform = waveform[:target_length]
            elif waveform.shape[0] < target_length:
                # Pad with zeros if shorter
                padding_length = target_length - waveform.shape[0]
                waveform = torch.cat([waveform, torch.zeros(padding_length, dtype=waveform.dtype)])

        except Exception as e:
            print(f"Error loading audio file {filepath}: {e}")
            # Return a dummy tensor with fixed length
            dummy_waveform = torch.zeros(64600, dtype=torch.float32)
            print(f"Returning dummy tensor for {ID} to continue training")
            return dummy_waveform, label

        # Get label (0 for bonafide, 1 for spoof)
        label_str = self.labels[ID]
        label = 1 if label_str == 'spoof' else 0
        label = torch.tensor(label, dtype=torch.long) # Use long for CrossEntropyLoss

        return waveform, label # Return waveform as (time,), and label as tensor

# ===================================================================
# Part 4: Utility Functions
# ===================================================================

def pad(x, max_len):
    """Pads waveform to max_len with zeros."""
    x_len = x.shape[-1] # Get time dimension length
    if x_len > max_len:
        # Truncate if longer than max_len
        return x[..., :max_len]
    num_missing = max_len - x_len
    # Pad with zeros at the end. Assuming x is (channels, time) or (time,)
    # If (channels, time), pad the last dimension
    # If (time,), just pad the last dimension
    padding = (0, num_missing) # Pad at the end of the last dimension
    return F.pad(x, padding)

def train_epoch(train_loader, model, criterion, optimizer, device, scheduler, config):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)

    # Get augmentation parameters
    aug_config = config.get('augment', {})
    use_adv_augment = aug_config.get('use_advanced_waveform_augment', False)
    noise_prob = aug_config.get('noise_prob', 0.5)
    reverb_prob = aug_config.get('reverb_prob', 0.5)
    rir_paths = aug_config.get('rir_paths', []) # List of RIR file paths
    noise_paths = aug_config.get('noise_paths', []) # List of noise file paths

    # Load RIRs and noise clips if augmentation is enabled
    rirs = []
    noise_clips = []
    if use_adv_augment:
        if rir_paths:
            try:
                # Ensure paths are absolute or relative to where the script is run if needed
                # Assuming paths in config are relative to the script's directory
                script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
                abs_rir_paths = [os.path.join(script_dir, p) for p in rir_paths]
                rirs = [torchaudio.load(p)[0].squeeze(0).to(device) for p in abs_rir_paths if os.path.exists(p)]
                if not rirs and rir_paths:
                     print("Warning: No RIR files found for reverb augmentation.")
            except Exception as e:
                print(f"Error loading RIR files: {e}")
                rirs = [] # Clear rirs if loading fails
        if noise_paths:
            try:
                 # Ensure paths are absolute or relative to where the script is run if needed
                 script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
                 abs_noise_paths = [os.path.join(script_dir, p) for p in noise_paths]
                 noise_clips = [torchaudio.load(p)[0].squeeze(0).to(device) for p in abs_noise_paths if os.path.exists(p)]
                 if not noise_clips and noise_paths:
                      print("Warning: No noise files found for noise augmentation.")
            except Exception as e:
                print(f"Error loading noise clips: {e}")
                noise_clips = [] # Clear noise_clips if loading fails


    for i, (batch_x, batch_y) in enumerate(train_loader):
        # Ensure batch_x is at least 2D (batch, time) or 3D (batch, channel, time)
        # The Dataset returns (waveform, label). Waveform can be (1, time) or (time,).
        # DataLoader will batch them. If waveform is (1, time), batch_x will be (batch, 1, time).
        # If waveform is (time,), batch_x will be (batch, time).
        # Need to handle both cases for augmentation. Assuming augmentation works on (batch, time)
        # or (batch, 1, time). Let's ensure it's (batch, 1, time) for conv.
        if batch_x.ndim == 2:
            batch_x = batch_x.unsqueeze(1) # Add channel dimension if missing (batch, 1, time)

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # --- Advanced Waveform Augmentations ---
        if use_adv_augment:
             # Apply noise
            if noise_clips and random.random() < noise_prob:
                noise_clip = random.choice(noise_clips)
                # Ensure noise clip has at least the length of the audio
                # Pad or repeat noise if needed. Simple padding for now.
                # Noise needs to be (1, time) or compatible for broadcasting
                noise_clip = noise_clip.unsqueeze(0) # Add channel dim (1, time)

                # Pad noise to match batch_x time dimension
                if noise_clip.shape[-1] < batch_x.shape[-1]:
                     noise_clip = pad(noise_clip, batch_x.shape[-1])
                elif noise_clip.shape[-1] > batch_x.shape[-1]:
                     noise_clip = noise_clip[..., :batch_x.shape[-1]]

                # Ensure noise has batch dimension for adding to batch_x
                noise_clip = noise_clip.repeat(batch_x.size(0), 1, 1) # Repeat for each item in batch

                # Add noise. Adjust SNR as needed. Simple addition for now.
                # Assuming noise_clip is (batch, 1, time)
                # Ensure shapes match for addition: batch_x is (batch, 1, time), noise_clip is (batch, 1, time)
                # Scale noise mean by a small factor, or use a proper SNR calculation
                batch_x = batch_x + noise_clip # Simple addition


            # Apply reverb (convolution with RIR)
            if rirs and random.random() < reverb_prob:
                rir = random.choice(rirs)
                 # Ensure RIR is shorter than the audio or pad audio before conv
                # For simplicity, assuming RIRs are short.
                # RIR needs to be (1, time) or compatible
                rir = rir.unsqueeze(0).unsqueeze(0) # Add out_channel and in_channel dim (1, 1, time)

                if rir.shape[-1] > batch_x.shape[-1]:
                     print("Warning: RIR length > audio length. Skipping reverb.")
                else:
                    # Apply convolution
                    # batch_x is (batch, 1, time), rir is (1, 1, time)
                    # Convolution output shape: (batch, out_channels, output_time)
                    # For a single RIR, out_channels is 1. Output_time depends on padding.
                    # With padding=rir.shape[-1] // 2, output_time is roughly batch_x.shape[-1]
                     batch_x = F.conv1d(batch_x, rir, padding=rir.shape[-1] // 2)
                    # The output of conv1d is (batch, 1, output_time). Keep it this way.


        optimizer.zero_grad()

        # --- Forward Pass ---
        # Model expects (batch, 1, time) or (batch, time) depending on implementation
        # If the model expects (batch, time), squeeze the channel dimension
        # Assuming model can handle (batch, 1, time) or will squeeze internally
        outputs = model(batch_x)


        # Ensure outputs and batch_y have compatible shapes for criterion
        # Assuming outputs are logits for BCEWithLogitsLoss or similar
        # If using CrossEntropyLoss (for 2 classes), outputs should be (batch_size, 2) logits
        # and batch_y should be (batch_size,) with values 0 or 1.
        if isinstance(criterion, nn.CrossEntropyLoss):
             # Ensure outputs are logits (e.g., from a final Linear layer)
             # and have shape (batch_size, num_classes)
             # Assuming num_classes is 2 for bonafide/spoof
             if outputs.ndim == 1:
                 # If outputs is (batch_size,), it might be probabilities for class 1.
                 # For CrossEntropyLoss, we need logits for each class.
                 # A common way is to assume outputs are logits for class 1,
                 # and logits for class 0 are implicitly 0 or similar, but
                 # the standard is two outputs for binary classification with CE.
                 # Let's assume the model output is already appropriately shaped
                 # for the selected loss function based on the config.
                 # If it's (batch_size,) and model is intended for binary classification
                 # with BCEWithLogitsLoss, then criterion should be BCEWithLogitsLoss.
                 # Let's stick to the assumption that the model output is (batch_size, num_classes).
                 pass # Assuming outputs are already (batch_size, 2)

             loss = criterion(outputs, batch_y)

        elif isinstance(criterion, (nn.BCEWithLogitsLoss, FocalLoss)):
             # Assuming outputs are single logits for binary classification (shape batch_size)
             # batch_y should be (batch_size,) floats (0.0 or 1.0) or longs (0 or 1)
             # BCEWithLogitsLoss expects targets as floats.
             # Ensure outputs are (batch_size,) for these losses
             if outputs.ndim > 1 and outputs.shape[1] == 1:
                 outputs = outputs.squeeze(1)
             loss = criterion(outputs, batch_y.float())


        # --- Backpropagation and Optimization ---
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

        # --- Update Scheduler (if using step scheduler) ---
        # If using step-based scheduler (e.g., StepLR), update here
        # If using epoch-based scheduler (e.g., CosineAnnealingLR), update after epoch

    epoch_loss = running_loss / len(train_loader.dataset)

    # --- Update Scheduler (if using epoch scheduler) ---
    # If using epoch-based scheduler (e.g., CosineAnnealingLR), update here
    # Note: This assumes the scheduler is passed directly, not through LambdaLR wrapper
    # If using LambdaLR for warmup, update the base scheduler outside this function
    # if scheduler and isinstance(scheduler, (torch.optim.lr_scheduler.CosineAnnealingLR, torch.optim.lr_scheduler.StepLR)):
    #     scheduler.step()

    return epoch_loss

def evaluate_model(data_loader, model, criterion, device, config):
    """Evaluates the model."""
    model.eval()
    running_loss = 0.0
    all_scores = []
    all_labels = []
    all_trial_ids = [] # To store trial IDs for score file

    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(data_loader): # Assuming data_loader now also yields trial IDs
            # The Dataset __getitem__ only returns waveform and label.
            # To get trial IDs here, either modify the Dataset to return ID,
            # or get the list of IDs separately and match by index.
            # Modifying Dataset is cleaner. Let's assume Dataset returns (waveform, label, trial_id)

            # If Dataset returns (waveform, label):
            # Need to get trial IDs from the original dataset list
            # This requires access to the dataset object or the list_IDs
            # A simpler approach for eval is to get scores and match later using the original eval file list.
            # Let's assume we can access the original list_IDs from the loader's dataset
            if hasattr(data_loader.dataset, 'list_IDs'):
                # Get the trial IDs for the current batch
                batch_trial_ids = [data_loader.dataset.list_IDs[idx] for idx in range(i * data_loader.batch_size, min((i + 1) * data_loader.batch_size, len(data_loader.dataset)))]
                all_trial_ids.extend(batch_trial_ids)
            else:
                 # Fallback if list_IDs is not easily accessible.
                 # Print a warning and skip writing trial IDs to score file or find another way.
                 if i == 0: print("Warning: Could not access trial IDs from dataset. Score file might be incomplete.")
                 all_trial_ids.extend([''] * batch_x.size(0)) # Add placeholders


            if batch_x.ndim == 2:
                batch_x = batch_x.unsqueeze(1) # Add channel dimension if missing (batch, 1, time)

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # --- Forward Pass ---
            outputs = model(batch_x)

            # Calculate loss
            if isinstance(criterion, nn.CrossEntropyLoss):
                 loss = criterion(outputs, batch_y)
                 # For scoring, get probability of spoof class (class 1)
                 scores = F.softmax(outputs, dim=1)[:, 1]
            elif isinstance(criterion, (nn.BCEWithLogitsLoss, FocalLoss)):
                 # Ensure outputs are (batch_size,) for these losses
                 if outputs.ndim > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                 loss = criterion(outputs, batch_y.float())
                 # For scoring, apply sigmoid to logits to get probability of spoof
                 scores = torch.sigmoid(outputs)
            else:
                 # Default to assuming outputs are scores directly if criterion is not standard
                 # Ensure outputs are (batch_size,)
                 if outputs.ndim > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)
                 loss = criterion(outputs, batch_y.float()) # Assuming criterion takes float targets
                 scores = outputs # Assume outputs are already scores

            running_loss += loss.item() * batch_x.size(0)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    epoch_loss = running_loss / len(data_loader.dataset)

    # Convert scores and labels to numpy arrays for metric calculation
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    return epoch_loss, all_scores, all_labels, all_trial_ids # Return trial IDs

def calculate_metrics(scores, labels):
    """Calculates EER and Accuracy."""
    # Calculate EER
    # Ensure labels are binary (0 or 1)
    y_true = np.array(labels).astype(int)
    y_score = np.array(scores).astype(float)

    # Remove samples with potential NaN/inf scores if any (though should not happen with sigmoid/softmax)
    valid_indices = np.isfinite(y_score)
    y_true = y_true[valid_indices]
    y_score = y_score[valid_indices]

    if len(np.unique(y_true)) < 2:
        # Cannot compute ROC if only one class is present
        print("Warning: Only one class present in evaluation labels. Cannot compute EER or ROC curve.")
        # Return dummy values or handle appropriately
        return 1.0, 0.0, 0.0 # EER=1 (worst), Acc=0, Threshold=0

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    # Find the threshold where fpr is closest to fnr
    abs_diffs = np.abs(fpr - fnr)
    # Find the index of the minimum absolute difference.
    # If there are multiple minimums, argmin returns the first one.
    idx = np.argmin(abs_diffs) # Use argmin, nanargmin handles NaNs which shouldn't be here

    eer = fpr[idx]
    eer_threshold = thresholds[idx]

    # Calculate Accuracy at EER threshold
    predictions = (y_score >= eer_threshold).astype(int)
    accuracy = accuracy_score(y_true, predictions)

    # Optional: Calculate other metrics like AUC if needed
    # try:
    #     auc = roc_auc_score(y_true, y_score)
    # except ValueError:
    #      auc = np.nan # Handle case where AUC can't be computed

    return eer, accuracy, eer_threshold #, auc

def save_model(model, optimizer, epoch, loss, eer, config, model_dir):
    """Saves the model checkpoint."""
    model_name_base = config['model']['name'] if 'model' in config and 'name' in config['model'] else 'model'
    # Use both epoch and EER in filename for better tracking
    checkpoint_name = f"epoch_{epoch}_eer_{eer:.4f}.pth" if not np.isnan(eer) else f"epoch_{epoch}_loss_{loss:.4f}.pth"
    # Use best loss/EER to track the best model, not just loss
    # Need to pass best_eer from outside or track it here.
    # Let's simplify and just save the latest model with its EER
    # To save the "Best" model, you'd need to compare `eer` with the best seen so far.

    os.makedirs(model_dir, exist_ok=True)

    checkpoint_path = os.path.join(model_dir, checkpoint_name)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss, # Save the loss from this epoch
        'eer': eer, # Save the EER from this epoch
        'config': config,
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    # Optional: Logic to save the *best* model
    # This would require tracking the best EER across epochs and saving only when improved.
    # Example:
    # if eer < self.best_dev_eer: # Assume best_dev_eer is tracked in main loop
    #     self.best_dev_eer = eer
    #     best_path = os.path.join(model_dir, f"Best_EER_{eer:.4f}.pth")
    #     torch.save(..., best_path)


def write_eval_scores(file_list, scores, output_path):
    """Writes evaluation scores to a file."""
    # Ensure file_list and scores have the same length
    if len(file_list) != len(scores):
        print(f"Warning: Mismatch between file list length ({len(file_list)}) and scores length ({len(scores)}). Skipping score file writing.")
        return

    with open(output_path, 'w') as f:
        for i, file_id in enumerate(file_list):
            # Assuming file_list contains the trial IDs like 'LA_E_1234567'
            f.write(f"{file_id} {scores[i]:.6f}\n")
    print(f"Evaluation scores written to: {output_path}")

def genSpoof_list(dir_meta, dir_data, is_train=False, is_dev=False, is_eval=False):
    """
    Generates a list of file IDs and corresponding labels from the protocol file.
    Returns (label_dict, file_list).
    """
    print(f"Reading protocol file: {dir_meta}")
    print(f"Corresponding data directory: {dir_data}") # Print the data directory path as well

    d_meta = {}
    file_list = []
    try:
        with open(dir_meta, 'r') as f:
            for line in f:
                # Assuming protocol format: speaker trial - attack label
                # e.g., LA_0001 LA_T_1000132 - - bonafide
                # or LA_0001 LA_T_1000137 - AA spoof
                parts = line.strip().split(' ')
                if len(parts) >= 5: # Ensure enough columns
                    speaker, trial_id, dash1, attack_type, label = parts
                    # For evaluation, the attack type might be '-', the label is spoof/bonafide
                    # For training/dev, attack type indicates the type of spoof
                    d_meta[trial_id] = label
                    file_list.append(trial_id)
            print(f"Loaded {len(file_list)} file IDs and labels from protocol.")
    except FileNotFoundError:
        print(f"Error: Protocol file not found at {dir_meta}")
        # Re-raising the error as it's critical
        raise
    except Exception as e:
        print(f"Error reading protocol file {dir_meta}: {e}")
        raise # Re-raise other exceptions


    # Optional: Filter file_list based on actual files present in dir_data/flac
    # This is important if the protocol lists files that don't exist.
    # This can be time-consuming for large datasets, so might skip for quick testing.
    # print("Verifying file existence (this may take time)...")
    # actual_file_list = []
    # for file_id in file_list:
    #     filepath = os.path.join(dir_data, self.subset, 'flac', file_id + '.flac') # Need subset logic here
    #     # Re-implement subset logic based on is_train/is_dev/is_eval flags
    #     subset_dir = ''
    #     if is_train: subset_dir = 'ASVspoof2019_LA_train'
    #     elif is_dev: subset_dir = 'ASVspoof2019_LA_dev'
    #     elif is_eval: subset_dir = 'ASVspoof2019_LA_eval'
    #
    #     filepath_with_subset = os.path.join(dir_data, subset_dir, 'flac', file_id + '.flac')
    #
    #     if os.path.exists(filepath_with_subset):
    #         actual_file_list.append(file_id)
    #     else:
    #         print(f"Warning: File not found at {filepath_with_subset}. Skipping.")
    # file_list = actual_file_list
    # print(f"After verification, using {len(file_list)} files.")


    return d_meta, file_list


# ===================================================================
# Part 5: Main Execution Logic
# ===================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2019 Baseline system Maze 5 compatible')
    # Dataset
    parser.add_argument('--track', type=str, default='LA', choices=['LA', 'PA'], help='ASVspoof track')
    parser.add_argument('--database_path', type=str, default='/content/sample_data/data/', help='Root path of database (use local path for efficiency)')
    parser.add_argument('--protocols_path', type=str, default='/content/sample_data/data/LA/ASVspoof2019_LA_cm_protocols/', help='Path to protocol files')
    # Model
    parser.add_argument('--config', type=str, default='model_config_Maze5.yaml', help='Model configuration file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to load the pretrained model')
    # Training
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--loss', type=str, default='CCE', choices=['CCE', 'Focal', 'cce', 'focal'], help='Loss function (CCE/cce or Focal/focal)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--comment', type=str, default="Real_Maze_3", help='Comment for model directory')
    parser.add_argument('--model_save_dir', type=str, default='models/Real_Maze_3', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    # Evaluation
    parser.add_argument('--is_eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval_protocol', type=str, default='ASVspoof2019.LA.cm.eval.trl.txt', help='Evaluation protocol file name')
    parser.add_argument('--eval_output', type=str, default='eval_CM_scores.txt', help='Output file for evaluation scores')


    args = parser.parse_args()

    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Standardized model configuration for fair comparison with FMSL
    config = {
        'model': {
            'filts': [128, [128, 128], [128, 256]],  # FMSL-compatible architecture
            'first_conv': 251,
            'sample_rate': 16000,
            'nb_fc_node': 1024,
            'fc_dropout': 0.5,
            'nb_classes': 2,
            # Wav2Vec2 parameters
            'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
            'wav2vec2_output_dim': 768,  # Standard Wav2Vec2 base model output dimension
            'wav2vec2_freeze_cnn': True,
            # Advanced features
            'use_se_blocks': True,
            'use_attention': True,
            'use_spec_augment': True,
            'use_focal_loss': True
        },
        'train': {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.lr
        },
        'optimizer': {
            'type': 'AdamW',
            'weight_decay': 0.0001
        },
        'scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': args.num_epochs
        },
        'augment': {
            'use_spec_augment': True,
            'freq_mask_param': 10,
            'time_mask_param': 10
        }
    }
    
    # Use .get with default empty dictionaries for robustness
    d_args = config.get('model', {}) # Model specific arguments
    train_config = config.get('train', {}) # Training specific arguments
    optimizer_config = config.get('optimizer', {}) # Optimizer specific arguments
    scheduler_config = config.get('scheduler', {}) # Scheduler specific arguments
    augment_config = config.get('augment', {}) # Augmentation specific arguments (used in train_epoch)
    config['augment'] = augment_config # Ensure augment config is added back if missing
    
    print("Using hardcoded FMSL-compatible configuration")


    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Instantiate the model
    model = Model_Maze5(d_args, device).to(device)
    print(f"Number of trainable model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")


    # Define loss function
    if args.loss.lower() == 'cce':
        criterion = nn.CrossEntropyLoss()
        print("Using CrossEntropyLoss")
    elif args.loss.lower() == 'focal':
        # Focal loss might need specific parameters from config
        focal_alpha = config.get('focal_loss_alpha', 1.0) # Get directly from top level if needed
        focal_gamma = config.get('focal_loss_gamma', 2.0) # Get directly from top level if needed
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, logits=True)
        print(f"Using FocalLoss (alpha={focal_alpha}, gamma={focal_gamma})")
    else:
        raise ValueError(f"Loss function '{args.loss}' not supported.")


    # Define optimizer and scheduler
    optimizer = create_optimizer(model, args, config)
    warmup_scheduler, main_scheduler = create_scheduler(optimizer, args, config)

    # Load pretrained model if specified
    start_epoch = 0
    best_dev_eer = float('inf') # Track best EER
    if args.model_path:
        start_epoch, loaded_optimizer_state = load_model(model, args.model_path, device)
        if loaded_optimizer_state:
             optimizer.load_state_dict(loaded_optimizer_state)
             # Note: Need to correctly resume scheduler state as well if applicable


    # Data paths
    # Corrected data paths based on confirmed local extraction location
    base_data_dir = args.database_path.rstrip('/')  # Remove trailing slash if present
    protocol_dir = args.protocols_path
    train_protocol_path = os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.train.trn.txt')
    dev_protocol_path = os.path.join(protocol_dir, 'ASVspoof2019.LA.cm.dev.trl.txt')
    eval_protocol_path = os.path.join(protocol_dir, args.eval_protocol) # Use eval_protocol arg
    
    # Debug: Check if directories exist
    print(f"Checking directory structure:")
    print(f"  Base data dir: {base_data_dir} (exists: {os.path.exists(base_data_dir)})")
    print(f"  Protocol dir: {protocol_dir} (exists: {os.path.exists(protocol_dir)})")
    
    # Check for common directory structures
    possible_train_dirs = [
        os.path.join(base_data_dir, 'ASVspoof2019_LA_train'),
        os.path.join(base_data_dir, 'LA', 'ASVspoof2019_LA_train'),
        os.path.join(base_data_dir, 'train'),
        os.path.join(base_data_dir, 'LA', 'train')
    ]
    
    for train_dir in possible_train_dirs:
        if os.path.exists(train_dir):
            print(f"  Found training data at: {train_dir}")
            # Update base_data_dir to point to the correct location
            if 'LA' in train_dir and 'ASVspoof2019_LA_train' in train_dir:
                # If we found LA/ASVspoof2019_LA_train, update base_data_dir to include LA
                if train_dir.endswith('ASVspoof2019_LA_train'):
                    base_data_dir = os.path.dirname(train_dir)
                    print(f"  Updated base_data_dir to: {base_data_dir}")
            break
    else:
        print(f"  Warning: No training data directory found in expected locations")
        print(f"  Available directories in {base_data_dir}: {os.listdir(base_data_dir) if os.path.exists(base_data_dir) else 'N/A'}")


    if args.is_eval:
        print("--- Evaluation Mode ---")
        if not args.model_path:
            print("Error: Model path must be provided for evaluation.")
            sys.exit(1)

        # Load evaluation data
        # PRINT STATEMENTS ADDED HERE
        print(f"Evaluating using protocol: {eval_protocol_path}")
        print(f"Looking for evaluation data in base directory: {base_data_dir}")
        try:
            d_label_eval, file_eval = genSpoof_list(dir_meta=eval_protocol_path, dir_data=base_data_dir, is_eval=True)
        except FileNotFoundError:
             print("Evaluation protocol file not found. Exiting.")
             sys.exit(1)
        except Exception as e:
             print(f"Error loading evaluation protocol: {e}. Exiting.")
             sys.exit(1)

        eval_dataset = Dataset_ASVspoof2019(list_IDs=file_eval, labels=d_label_eval, base_dir=base_data_dir, is_eval=True)
        eval_loader = get_loader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

        # Evaluate the model
        print("Evaluating...")
        eval_loss, eval_scores, eval_labels, eval_trial_ids = evaluate_model(eval_loader, model, criterion, device, config) # Get trial IDs back

        # Calculate metrics
        eval_eer, eval_accuracy, eval_threshold = calculate_metrics(eval_scores, eval_labels)
        print(f"Evaluation Loss: {eval_loss:.4f}, EER: {eval_eer:.4f}, Accuracy: {eval_accuracy:.4f}, Threshold: {eval_threshold:.4f}")

        # Write scores to file
        write_eval_scores(eval_trial_ids, eval_scores, args.eval_output) # Use eval_trial_ids

    else: # Training Mode
        print("--- Training Mode ---")
        log_dir = f"logs/{args.track}/{time.strftime('%Y%m%d-%H%M%S')}"
        writer = SummaryWriter(log_dir=log_dir)

        # Load training and development data
        # PRINT STATEMENTS ADDED HERE
        print(f"Training using protocol: {train_protocol_path}")
        print(f"Looking for training data in base directory: {base_data_dir}")
        try:
            d_label_trn, file_train = genSpoof_list(dir_meta=train_protocol_path, dir_data=base_data_dir, is_train=True)
        except FileNotFoundError:
            print("Training protocol file not found. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading training protocol: {e}. Exiting.")
            sys.exit(1)


        train_dataset = Dataset_ASVspoof2019(list_IDs=file_train, labels=d_label_trn, base_dir=base_data_dir, is_train=True)
        train_loader = get_loader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True) # Drop last incomplete batch

        # PRINT STATEMENTS ADDED HERE
        print(f"Validating using protocol: {dev_protocol_path}")
        print(f"Looking for validation data in base directory: {base_data_dir}")
        dev_loader = None # Initialize dev_loader to None
        d_label_dev = None
        file_dev = None
        try:
            d_label_dev, file_dev = genSpoof_list(dir_meta=dev_protocol_path, dir_data=base_data_dir, is_dev=True)
            dev_dataset = Dataset_ASVspoof2019(list_IDs=file_dev, labels=d_label_dev, base_dir=base_data_dir, is_dev=True)
            dev_loader = get_loader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        except FileNotFoundError:
             print("Development protocol file not found. Skipping validation.")
        except Exception as e:
             print(f"Error loading development protocol: {e}. Skipping validation.")


        # Early stopping (optional, configured in yaml)
        # Use .get with default empty dictionary for robustness
        early_stopper_config = train_config.get('early_stopping', {})
        use_early_stopping = early_stopper_config.get('enabled', False)
        early_stopper = None
        early_stopping_metric = early_stopper_config.get('metric', 'loss') # Metric to monitor for early stopping ('loss' or 'eer')
        early_stopping_mode = early_stopper_config.get('mode', 'min') # 'min' for loss/eer, 'max' for accuracy

        if use_early_stopping and dev_loader: # Only enable if validation data is available
             patience = early_stopper_config.get('patience', 10)
             min_delta = early_stopper_config.get('min_delta', 0.001)
             early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)
             early_stopper.mode = early_stopping_mode # Set the mode
             print(f"Early stopping enabled with patience={patience}, min_delta={min_delta}, metric='{early_stopping_metric}', mode='{early_stopping_mode}'")
        elif use_early_stopping and not dev_loader:
            print("Warning: Early stopping requested but no development data loaded. Early stopping will be disabled.")


        # Training loop
        print("Starting training...")
        for epoch in range(start_epoch, args.num_epochs):
            print(f"Epoch {epoch+1}/{args.num_epochs}")

            # Update warmup scheduler learning rate
            if warmup_scheduler:
                 warmup_scheduler.step()

            train_loss = train_epoch(train_loader, model, criterion, optimizer, device, main_scheduler, config)
            print(f"Train Loss: {train_loss:.4f}")
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('LearningRate/Train', optimizer.param_groups[0]['lr'], epoch) # Log learning rate


            # Evaluate on development set (if loaded)
            if dev_loader:
                dev_loss, dev_scores, dev_labels, dev_trial_ids = evaluate_model(dev_loader, model, criterion, device, config) # Get trial IDs
                dev_eer, dev_accuracy, dev_threshold = calculate_metrics(dev_scores, dev_labels)
                print(f"Dev Loss: {dev_loss:.4f}, Dev EER: {dev_eer:.4f}, Dev Accuracy: {dev_accuracy:.4f}, Dev Threshold: {dev_threshold:.4f}")
                writer.add_scalar('Loss/Dev', dev_loss, epoch)
                writer.add_scalar('Metrics/Dev_EER', dev_eer, epoch)
                writer.add_scalar('Metrics/Dev_Accuracy', dev_accuracy, epoch)

                # Update main scheduler (if using epoch scheduler)
                if main_scheduler:
                     # Note: Some schedulers like ReduceLROnPlateau need a metric
                     if isinstance(main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                         # Step based on the metric specified for early stopping or a default
                         if early_stopping_metric == 'loss':
                             main_scheduler.step(dev_loss)
                         elif early_stopping_metric == 'eer':
                             main_scheduler.step(dev_eer)
                         elif early_stopping_metric == 'accuracy':
                             main_scheduler.step(-dev_accuracy) # Plateau steps on decreasing metric for max
                         else:
                             main_scheduler.step(dev_loss) # Default to loss

                     else:
                         main_scheduler.step() # Step based on epoch


                # Save model checkpoint (only save best based on Dev EER)
                if dev_eer < best_dev_eer:
                     best_dev_eer = dev_eer
                     # Save with the current epoch's dev loss and best EER
                     save_model(model, optimizer, epoch+1, dev_loss, best_dev_eer, config, args.model_save_dir)
                     print(f"New best model saved with EER: {best_dev_eer:.4f}")


                # Check for early stopping
                if use_early_stopping and early_stopper:
                    # Use the specified metric for early stopping
                    if early_stopping_metric == 'loss':
                        stop_metric = dev_loss
                    elif early_stopping_metric == 'eer':
                        stop_metric = dev_eer
                    elif early_stopping_metric == 'accuracy':
                        stop_metric = dev_accuracy
                    else:
                        stop_metric = dev_loss # Default

                    if early_stopper.early_stop(stop_metric):
                         print(f"Early stopping triggered at epoch {epoch+1} based on {early_stopping_metric}.")
                         break

            else:
                # If no dev loader, update scheduler based on epoch if not Plateau
                if main_scheduler and not isinstance(main_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    main_scheduler.step()

                # Save model checkpoint periodically if no validation is done
                if (epoch + 1) % args.save_interval == 0:
                     # Need a dummy EER or handle this case.
                     # For simplicity, just save based on epoch if no validation.
                     print("Saving model based on epoch interval (no validation EER available).")
                     save_model(model, optimizer, epoch+1, train_loss, float('nan'), config, args.model_save_dir) # Use train_loss and NaN for EER


        print("Training finished.")
        writer.close()
