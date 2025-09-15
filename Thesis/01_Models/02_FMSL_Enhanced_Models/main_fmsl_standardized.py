#!/usr/bin/env python3
# ===================================================================
# main_fmsl_standardized.py - Main RawNet Model + Standardized FMSL
# 
# DESCRIPTION:
# This is the STANDARDIZED version of main.py with proper FMSL implementation
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
import numpy as np
import torch
import yaml
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from maze import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from maze import RawNet
from tensorboardX import SummaryWriter
# Local definition of set_random_seed function
def set_random_seed(seed, args):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.cudnn_deterministic_toggle:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = args.cudnn_benchmark_toggle

# Import the standardized FMSL system and configuration
from fmsl_advanced import AdvancedFMSLSystem, create_fmsl_config
from fmsl_standardized_config import get_standardized_model_config

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"
__credits__ = ["Jose Patino", "Massimiliano Todisco", "Jee-weon Jung"]

# ===================================================================
# Part 1: Enhanced RawNet Model with FMSL
# ===================================================================

class RawNetWithFMSL(nn.Module):
    """
    Enhanced RawNet model with standardized FMSL system
    """
    def __init__(self, model_config, device):
        super(RawNetWithFMSL, self).__init__()
        self.device = device
        
        # Original RawNet backbone
        self.backbone = RawNet(model_config, device)
        
        # Get the output dimension from the backbone
        # We need to determine this dynamically or from config
        backbone_output_dim = model_config.get('backbone_output_dim', 512)
        
        # ✅ NEW: Replace classifier with standardized FMSL system
        fmsl_config = create_fmsl_config(
            model_type=model_config.get('fmsl_type', 'prototype'),
            n_prototypes=model_config.get('fmsl_n_prototypes', 3),
            s=model_config.get('fmsl_s', 32.0),
            m=model_config.get('fmsl_m', 0.45),
            enable_lsa=model_config.get('fmsl_enable_lsa', True)
        )
        
        # Filter out any unsupported parameters that might be in the config
        supported_params = {
            'input_dim': backbone_output_dim,
            'n_classes': 2,  # Binary classification
            'use_integrated_loss': False,  # Use CCE instead of integrated FMSL loss
            'use_prototypes': fmsl_config.get('use_prototypes', True),
            'n_prototypes': fmsl_config.get('n_prototypes', 3),
            's': fmsl_config.get('s', 32.0),
            'm': fmsl_config.get('m', 0.45),
            'enable_lsa': fmsl_config.get('enable_lsa', True)
        }
        
        # Initialize FMSL system with CCE loss for consistency
        self.fmsl_system = AdvancedFMSLSystem(**supported_params)
        
        # Add CCE loss function for consistency with normal maze models
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))
    
    def forward(self, x, labels=None, training=False):
        # Extract features through RawNet backbone (before final classification)
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)
        
        # Go through RawNet layers up to the GRU output
        x = self.backbone.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.backbone.first_bn(x)
        x = self.backbone.selu(x)
        
        x0 = self.backbone.block0(x)
        y0 = self.backbone.avgpool(x0).view(x0.size(0), -1)
        y0 = self.backbone.fc_attention0(y0)
        y0 = self.backbone.sig(y0).view(y0.size(0), y0.size(1), -1)
        x = x0 * y0 + y0
        
        x1 = self.backbone.block1(x)
        y1 = self.backbone.avgpool(x1).view(x1.size(0), -1)
        y1 = self.backbone.fc_attention1(y1)
        y1 = self.backbone.sig(y1).view(y1.size(0), y1.size(1), -1)
        x = x1 * y1 + y1
        
        x2 = self.backbone.block2(x)
        y2 = self.backbone.avgpool(x2).view(x2.size(0), -1)
        y2 = self.backbone.fc_attention2(y2)
        y2 = self.backbone.sig(y2).view(y2.size(0), y2.size(1), -1)
        x = x2 * y2 + y2
        
        x3 = self.backbone.block3(x)
        y3 = self.backbone.avgpool(x3).view(x3.size(0), -1)
        y3 = self.backbone.fc_attention3(y3)
        y3 = self.backbone.sig(y3).view(y3.size(0), y3.size(1), -1)
        x = x3 * y3 + y3
        
        x4 = self.backbone.block4(x)
        y4 = self.backbone.avgpool(x4).view(x4.size(0), -1)
        y4 = self.backbone.fc_attention4(y4)
        y4 = self.backbone.sig(y4).view(y4.size(0), y4.size(1), -1)
        x = x4 * y4 + y4
        
        x5 = self.backbone.block5(x)
        y5 = self.backbone.avgpool(x5).view(x5.size(0), -1)
        y5 = self.backbone.fc_attention5(y5)
        y5 = self.backbone.sig(y5).view(y5.size(0), y5.size(1), -1)
        x = x5 * y5 + y5
        
        x = self.backbone.bn_before_gru(x)
        x = self.backbone.selu(x)
        x = x.permute(0, 2, 1)
        self.backbone.gru.flatten_parameters()
        x, _ = self.backbone.gru(x)
        x = x[:, -1, :]  # Get the last GRU output
        
        # Extract features before final classification
        features = self.backbone.fc1_gru(x)  # This outputs 1024 features
        
        # Apply FMSL geometric shaping (without integrated loss)
        fmsl_output = self.fmsl_system(features, labels, training=training)
        
        # Use CCE loss for consistency with normal maze models
        if training and labels is not None:
            loss = self.criterion(fmsl_output['logits'], labels)
            return {
                'logits': fmsl_output['logits'],
                'loss': loss,
                'features': fmsl_output['normalized_embeddings']
            }
        else:
            return {
                'logits': fmsl_output['logits'],
                'features': fmsl_output['normalized_embeddings']
            }

# ===================================================================
# Part 2: Training and Evaluation Functions (Updated for FMSL)
# ===================================================================

def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # ✅ NEW: No labels needed for evaluation
        batch_out = model(batch_x, training=False)
        batch_logits = batch_out['logits']
        
        _, batch_pred = batch_logits.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    
    for batch_x, utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        # ✅ NEW: No labels needed for evaluation
        batch_out = model(batch_x, training=False)
        batch_logits = batch_out['logits']
        
        batch_score = (batch_logits[:, 1]).data.cpu().numpy().ravel()
        
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    # ✅ REMOVED: No need for separate loss function - FMSL system handles it
    
    for batch_x, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        
        # ✅ NEW: Pass labels to model for FMSL loss computation
        batch_out = model(batch_x, batch_y, training=True)
        batch_loss = batch_out['loss']  # Loss computed by CCE
        batch_logits = batch_out['logits']
        
        _, batch_pred = batch_logits.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        
        if ii % 10 == 0:
            sys.stdout.write('\r \t {:.2f}'.format(
                (num_correct/num_total)*100))
        
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

# ===================================================================
# Part 3: Main Training Script
# ===================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system with Standardized FMSL')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac
    '''

    parser.add_argument('--protocols_path', type=str, default='/content/drive/MyDrive/ASVspoof2019/Extract/LA/ASVspoof2019_LA_cm_protocols/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt 
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Default Model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 
    

    # No YAML file needed - using standardized configuration system
    # dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    # with open(dir_yaml, 'r') as f_yaml:
    #         parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    if not os.path.exists('models'):
        os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF'], 'Invalid track given'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    #define model saving path
    model_tag = 'main_fmsl_standardized_{}_{}_{}_{}_{}'.format(
        track, args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    # ✅ NEW: Use standardized configuration
    model_config = get_standardized_model_config('main')
    
    # Update with RawNet-specific configuration
    model_config.update({
        'backbone_output_dim': 1024,  # Fixed: Use the actual output dimension from fc1_gru
        'fmsl_type': 'prototype',
        'fmsl_n_prototypes': 3,
        'fmsl_s': 32.0,
        'fmsl_m': 0.45,
        'fmsl_enable_lsa': True
    })
    
    #model 
    model = RawNetWithFMSL(model_config, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    
    print(f'Trainable model parameters: {nb_params / 1e6:.2f} M')
    print(f'FMSL system parameters: {sum(p.numel() for p in model.fmsl_system.parameters()):,}')
    
    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
        print('Model loaded : {}'.format(args.model_path))

    # ✅ UPDATED: Using CCE loss for consistency with normal maze models
    print(f"Loss function: Categorical Cross-Entropy (CCE) with FMSL geometric features")

    #evaluation 
    if args.eval:
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'{}.cm.eval.trl.txt'.format('ASVspoof2019.LA')),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'ASVspoof2019_{}_eval/'.format(args.track)))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

     
    # define train dataloader

    d_label_trn,file_train = genSpoof_list( dir_meta = os.path.join(args.protocols_path, "{}.cm.train.trn.txt".format('ASVspoof2019.LA')),is_train=True,is_eval=False)
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train(list_IDs = file_train,labels = d_label_trn,base_dir = os.path.join(args.database_path+'ASVspoof2019_{}_train/'.format(args.track)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list( dir_meta = os.path.join(args.protocols_path, "{}.cm.dev.trl.txt".format('ASVspoof2019.LA')),is_train=False,is_eval=False)
    print('no. of validation trials',len(file_dev))

    dev_set = Dataset_ASVspoof2019_train(list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = os.path.join(args.database_path+'ASVspoof2019_{}_dev/'.format(args.track)))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)
    del dev_set,d_label_dev

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    best_acc = 0  # Changed from 99 to 0 for proper comparison
    best_epoch = 0
    
    for epoch in range(num_epochs):
        running_loss, train_accuracy = train_epoch(train_loader,model, args.lr,optimizer, device)
        valid_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_accuracy, valid_accuracy))
        
        # Save every epoch
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch+1}_{model_tag}.pth'))
        
        # Track best model
        if valid_accuracy > best_acc:
            print('best model found at epoch', epoch+1)
            best_acc = valid_accuracy
            best_epoch = epoch + 1
            
            # Remove previous best model files
            for f in os.listdir(model_save_path):
                if f.startswith('best_epoch_'):
                    os.remove(os.path.join(model_save_path, f))
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(model_save_path, f'best_epoch_{epoch+1}_{model_tag}.pth'))
    
    print(f'\nTraining completed! Best model (epoch {best_epoch}) saved to: {model_save_path}')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'✅ STANDARDIZED FMSL implementation completed!')
    print(f'✅ True geometric manifold shaping with angular margins implemented!')
    print(f'✅ CRITICAL FIX: Using CCE loss for consistency with normal maze models!')
    print(f'✅ Model saved to: {model_save_path}')
    writer.close()
