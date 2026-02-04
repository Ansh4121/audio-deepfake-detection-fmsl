#!/usr/bin/env python3
"""
Advanced FMSL (Feature Matching Self-Supervised Learning) System
================================================================

This module implements the core FMSL system used by all FMSL-enhanced models
in this thesis. It provides geometric feature manifold shaping for improved
audio deepfake detection.

Key Components:
- AdvancedFMSLSystem: Main FMSL layer with prototype-based classification
- create_fmsl_config: Factory function for creating FMSL configurations

The FMSL approach addresses the geometric bottleneck identified in baseline
models by:
1. L2 normalization for hypersphere projection
2. Angular margin learning with AM-Softmax loss
3. Prototype-based classification for spoof class modeling
4. Latent space augmentation for improved generalization

Reference: Chapter 5 of the thesis for theoretical background.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Any


def create_fmsl_config(
    model_type: str = 'prototype',
    n_prototypes: int = 3,
    s: float = 32.0,
    m: float = 0.45,
    enable_lsa: bool = False,
    lsa_strength: float = 0.1,
    **kwargs
) -> Dict[str, Any]:
    """
    Create FMSL configuration dictionary.
    
    Args:
        model_type: Type of FMSL model ('prototype' for prototype-based)
        n_prototypes: Number of prototypes for spoof class modeling
        s: Scale factor for angular margin (AM-Softmax)
        m: Angular margin value
        enable_lsa: Enable latent space augmentation
        lsa_strength: Strength of latent space augmentation
        **kwargs: Additional configuration options
    
    Returns:
        dict: Complete FMSL configuration
        
    Example:
        >>> config = create_fmsl_config(n_prototypes=3, s=32.0, m=0.45)
        >>> fmsl = AdvancedFMSLSystem(input_dim=1024, n_classes=2, **config)
    """
    config = {
        'model_type': model_type,
        'n_prototypes': n_prototypes,
        's': s,
        'm': m,
        'enable_lsa': enable_lsa,
        'lsa_strength': lsa_strength,
    }
    config.update(kwargs)
    return config


class AdvancedFMSLSystem(nn.Module):
    """
    Advanced Feature Matching Self-Supervised Learning System.
    
    This module implements geometric feature manifold shaping for improved
    audio deepfake detection. It projects features onto a hypersphere and
    applies angular margin learning for better class separation.
    
    Architecture:
    1. Feature projection layer (optional)
    2. L2 normalization (hypersphere projection)
    3. Prototype-based spoof class modeling
    4. Angular margin loss computation
    
    Args:
        input_dim: Dimension of input features
        n_classes: Number of output classes (typically 2: bonafide/spoof)
        n_prototypes: Number of prototypes for spoof class
        s: Scale factor for AM-Softmax
        m: Angular margin value
        use_integrated_loss: Whether to use integrated FMSL loss
        enable_lsa: Enable latent space augmentation
        lsa_strength: Strength of latent space augmentation
        model_type: Type of FMSL model
        **kwargs: Additional arguments
        
    Example:
        >>> fmsl = AdvancedFMSLSystem(input_dim=1024, n_classes=2, n_prototypes=3)
        >>> output = fmsl(features, labels, training=True)
        >>> embeddings = output['normalized_embeddings']
    """
    
    def __init__(
        self,
        input_dim: int,
        n_classes: int = 2,
        n_prototypes: int = 3,
        s: float = 32.0,
        m: float = 0.45,
        use_integrated_loss: bool = False,
        enable_lsa: bool = False,
        lsa_strength: float = 0.1,
        model_type: str = 'prototype',
        **kwargs
    ):
        super(AdvancedFMSLSystem, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.s = s  # Scale factor for AM-Softmax
        self.m = m  # Angular margin
        self.use_integrated_loss = use_integrated_loss
        self.enable_lsa = enable_lsa
        self.lsa_strength = lsa_strength
        self.model_type = model_type
        
        # Feature projection to embedding space
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Prototype embeddings for spoof class modeling
        # Each prototype represents a cluster in the spoof manifold
        self.prototypes = nn.Parameter(
            torch.randn(n_prototypes, input_dim) * 0.01
        )
        nn.init.xavier_uniform_(self.prototypes)
        
        # Class weight matrix for angular margin
        self.weight = nn.Parameter(
            torch.randn(n_classes, input_dim) * 0.01
        )
        nn.init.xavier_uniform_(self.weight)
        
        # Learnable temperature for prototype matching
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def l2_normalize(self, x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
        """
        L2 normalize features for hypersphere projection.
        
        Args:
            x: Input tensor
            dim: Dimension along which to normalize
            eps: Small constant for numerical stability
            
        Returns:
            L2-normalized tensor
        """
        return F.normalize(x, p=2, dim=dim, eps=eps)
    
    def compute_prototype_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between features and spoof prototypes.
        
        This implements the prototype-based spoof class modeling where
        each spoof sample is matched against multiple learned prototypes.
        
        Args:
            features: Normalized feature embeddings [batch_size, input_dim]
            
        Returns:
            Prototype similarity scores [batch_size, n_prototypes]
        """
        # Normalize prototypes
        normalized_prototypes = self.l2_normalize(self.prototypes)
        
        # Compute cosine similarity
        similarity = torch.matmul(features, normalized_prototypes.t())
        
        # Scale by temperature
        similarity = similarity / self.temperature.clamp(min=0.01)
        
        return similarity
    
    def compute_angular_margin_logits(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = False
    ) -> torch.Tensor:
        """
        Compute logits with angular margin (AM-Softmax).
        
        The angular margin encourages larger angular separation between
        classes in the hypersphere feature space.
        
        Args:
            features: Normalized feature embeddings
            labels: Ground truth labels (required for margin during training)
            training: Whether in training mode
            
        Returns:
            Logits with angular margin applied
        """
        # Normalize weight matrix
        normalized_weight = self.l2_normalize(self.weight)
        
        # Compute cosine similarity (dot product of normalized vectors)
        cosine = torch.matmul(features, normalized_weight.t())
        
        if training and labels is not None:
            # Apply angular margin to correct class
            # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
            sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, max=1.0))
            cos_m = np.cos(self.m)
            sin_m = np.sin(self.m)
            
            # cos(theta + m)
            phi = cosine * cos_m - sine * sin_m
            
            # Create one-hot labels
            one_hot = F.one_hot(labels, num_classes=self.n_classes).float()
            
            # Apply margin only to correct class
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else:
            output = cosine
        
        # Scale output
        output = self.s * output
        
        return output
    
    def latent_space_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply latent space augmentation for improved generalization.
        
        Adds small random perturbations to features during training
        to improve robustness to unseen attack types.
        
        Args:
            features: Input features
            
        Returns:
            Augmented features
        """
        if self.training and self.enable_lsa:
            noise = torch.randn_like(features) * self.lsa_strength
            features = features + noise
        return features
    
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the FMSL system.
        
        Args:
            x: Input features [batch_size, input_dim]
            labels: Ground truth labels (optional, required for training)
            training: Whether in training mode
            
        Returns:
            Dictionary containing:
            - 'normalized_embeddings': L2-normalized feature embeddings
            - 'logits': Classification logits (if use_integrated_loss)
            - 'prototype_similarity': Similarity to spoof prototypes
            - 'features': Projected features before normalization
        """
        # Project features
        projected = self.projection(x)
        
        # Apply latent space augmentation
        projected = self.latent_space_augmentation(projected)
        
        # L2 normalize for hypersphere projection
        normalized = self.l2_normalize(projected)
        
        # Compute prototype similarity
        proto_sim = self.compute_prototype_similarity(normalized)
        
        # Prepare output dictionary
        output = {
            'normalized_embeddings': normalized,
            'features': projected,
            'prototype_similarity': proto_sim,
        }
        
        # Compute angular margin logits if using integrated loss
        if self.use_integrated_loss:
            logits = self.compute_angular_margin_logits(
                normalized, labels, training
            )
            output['logits'] = logits
        
        return output
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get normalized embeddings for inference.
        
        Args:
            x: Input features
            
        Returns:
            L2-normalized embeddings
        """
        projected = self.projection(x)
        normalized = self.l2_normalize(projected)
        return normalized
    
    def compute_fmsl_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute FMSL loss combining angular margin and prototype matching.
        
        Args:
            features: Normalized feature embeddings
            labels: Ground truth labels
            logits: Pre-computed logits (optional)
            
        Returns:
            Combined FMSL loss
        """
        if logits is None:
            logits = self.compute_angular_margin_logits(
                features, labels, training=True
            )
        
        # Cross-entropy with angular margin
        ce_loss = F.cross_entropy(logits, labels)
        
        # Prototype matching loss for spoof samples
        spoof_mask = (labels == 0).float()  # Assuming 0 is spoof
        if spoof_mask.sum() > 0:
            proto_sim = self.compute_prototype_similarity(features)
            # Maximize similarity to closest prototype
            max_sim, _ = proto_sim.max(dim=1)
            proto_loss = -spoof_mask * max_sim
            proto_loss = proto_loss.sum() / (spoof_mask.sum() + 1e-8)
        else:
            proto_loss = torch.tensor(0.0, device=features.device)
        
        # Combined loss
        total_loss = ce_loss + 0.1 * proto_loss
        
        return total_loss


class FMSLStandardizedConfig:
    """
    Standardized FMSL configuration used across all maze models.
    
    This ensures fair comparison between baseline and FMSL-enhanced
    models by using identical FMSL parameters.
    """
    
    # Default FMSL parameters (from standardized_maze_config.py)
    DEFAULT_CONFIG = {
        'model_type': 'prototype',
        'n_prototypes': 3,
        's': 32.0,
        'm': 0.45,
        'enable_lsa': False,
        'lsa_strength': 0.1
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get the standardized FMSL configuration."""
        return cls.DEFAULT_CONFIG.copy()
    
    @classmethod
    def create_fmsl_system(
        cls,
        input_dim: int,
        n_classes: int = 2,
        use_integrated_loss: bool = False
    ) -> AdvancedFMSLSystem:
        """
        Create an FMSL system with standardized configuration.
        
        Args:
            input_dim: Input feature dimension
            n_classes: Number of output classes
            use_integrated_loss: Whether to use integrated FMSL loss
            
        Returns:
            Configured AdvancedFMSLSystem instance
        """
        config = cls.get_config()
        return AdvancedFMSLSystem(
            input_dim=input_dim,
            n_classes=n_classes,
            use_integrated_loss=use_integrated_loss,
            **config
        )


# Convenience function for getting standardized model config
def get_standardized_model_config(model_type: str = "fmsl") -> Dict[str, Any]:
    """
    Get standardized model configuration.
    
    This function provides the same configuration used by all maze models
    to ensure fair comparison.
    
    Args:
        model_type: "baseline" or "fmsl"
        
    Returns:
        Configuration dictionary
    """
    config = {
        'architecture': {
            'filts': [128, [128, 128], [128, 256]],
            'nb_fc_node': 1024,
            'nb_classes': 2,
            'sample_rate': 16000,
            'first_conv': 251,
            'dropout_rate': 0.3
        },
        'wav2vec2': {
            'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
            'wav2vec2_output_dim': 768,
            'wav2vec2_freeze': True
        },
        'training': {
            'batch_size': 12,
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'grad_clip_norm': 1.0,
            'num_epochs': 5,
            'seed': 1234
        }
    }
    
    if model_type == "fmsl":
        config['fmsl'] = FMSLStandardizedConfig.get_config()
    
    # Flatten config for direct use
    flat_config = {}
    for category in config.values():
        if isinstance(category, dict):
            flat_config.update(category)
    
    return flat_config


if __name__ == "__main__":
    # Example usage and testing
    print("="*60)
    print("FMSL Advanced System - Test")
    print("="*60)
    
    # Create configuration
    config = create_fmsl_config(
        n_prototypes=3,
        s=32.0,
        m=0.45,
        enable_lsa=False
    )
    print(f"\nConfiguration: {config}")
    
    # Create FMSL system
    fmsl = AdvancedFMSLSystem(
        input_dim=1024,
        n_classes=2,
        use_integrated_loss=True,
        **config
    )
    print(f"\nFMSL System created with {sum(p.numel() for p in fmsl.parameters())} parameters")
    
    # Test forward pass
    batch_size = 4
    features = torch.randn(batch_size, 1024)
    labels = torch.randint(0, 2, (batch_size,))
    
    fmsl.train()
    output = fmsl(features, labels, training=True)
    
    print(f"\nOutput keys: {list(output.keys())}")
    print(f"Normalized embeddings shape: {output['normalized_embeddings'].shape}")
    print(f"Prototype similarity shape: {output['prototype_similarity'].shape}")
    
    if 'logits' in output:
        print(f"Logits shape: {output['logits'].shape}")
    
    print("\n FMSL System test passed!")
