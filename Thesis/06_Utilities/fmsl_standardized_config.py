#!/usr/bin/env python3
"""
FMSL Standardized Configuration
===============================

This module provides standardized configuration for FMSL-enhanced models.
It is imported by all maze*_fmsl_standardized.py files to ensure
consistent parameters across all models.

The configuration matches the settings used for thesis results as documented
in Chapter 6 (Experimental Validation).
"""

from typing import Dict, Any


def get_standardized_model_config(model_type: str = "fmsl") -> Dict[str, Any]:
    """
    Get standardized model configuration for fair comparison.
    
    This function returns the exact configuration used for all thesis
    results, ensuring reproducibility and fair comparison between
    baseline and FMSL-enhanced models.
    
    Args:
        model_type: "baseline" or "fmsl"
        
    Returns:
        dict: Complete model configuration
        
    Example:
        >>> config = get_standardized_model_config("fmsl")
        >>> print(config['nb_fc_node'])  # 1024
    """
    # Base architecture configuration (same for baseline and FMSL)
    config = {
        # Architecture parameters
        'filts': [128, [128, 128], [128, 256]],
        'nb_fc_node': 1024,
        'nb_classes': 2,
        'sample_rate': 16000,
        'first_conv': 251,
        'dropout_rate': 0.3,
        'fc_dropout': 0.5,
        
        # Wav2Vec2 parameters
        'wav2vec2_model_name': 'facebook/wav2vec2-base-960h',
        'wav2vec2_output_dim': 768,
        'wav2vec2_freeze': True,
        
        # Training parameters
        'batch_size': 12,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'grad_clip_norm': 1.0,
        'num_epochs': 5,
        'seed': 1234,
        
        # SpecAugment parameters
        'use_spec_augment_raw': True,
        'spec_aug_freq_mask_param_raw': 10,
        'spec_aug_time_mask_param_raw': 10,
        'spec_aug_n_freq_masks_raw': 2,
        'spec_aug_n_time_masks_raw': 2,
    }
    
    # Add FMSL-specific parameters
    if model_type == "fmsl":
        config.update({
            # FMSL parameters (as documented in Chapter 5)
            'fmsl_type': 'prototype',
            'fmsl_n_prototypes': 3,
            'fmsl_s': 32.0,  # Scale factor for AM-Softmax
            'fmsl_m': 0.45,  # Angular margin
            'fmsl_enable_lsa': False,  # Latent space augmentation
            'fmsl_lsa_strength': 0.1,
        })
    
    return config


def get_architecture_config() -> Dict[str, Any]:
    """
    Get architecture-only configuration.
    
    Returns:
        dict: Architecture parameters
    """
    return {
        'filts': [128, [128, 128], [128, 256]],
        'nb_fc_node': 1024,
        'nb_classes': 2,
        'sample_rate': 16000,
        'first_conv': 251,
        'dropout_rate': 0.3,
    }


def get_fmsl_config() -> Dict[str, Any]:
    """
    Get FMSL-only configuration.
    
    Returns:
        dict: FMSL parameters
    """
    return {
        'fmsl_type': 'prototype',
        'fmsl_n_prototypes': 3,
        'fmsl_s': 32.0,
        'fmsl_m': 0.45,
        'fmsl_enable_lsa': False,
        'fmsl_lsa_strength': 0.1,
    }


def get_training_config() -> Dict[str, Any]:
    """
    Get training-only configuration.
    
    Returns:
        dict: Training parameters
    """
    return {
        'batch_size': 12,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'grad_clip_norm': 1.0,
        'num_epochs': 5,
        'seed': 1234,
    }


def verify_config_consistency(config1: Dict, config2: Dict) -> bool:
    """
    Verify that two configurations have consistent architecture parameters.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        bool: True if consistent, False otherwise
    """
    architecture_keys = ['filts', 'nb_fc_node', 'nb_classes', 'sample_rate', 'first_conv']
    
    for key in architecture_keys:
        if config1.get(key) != config2.get(key):
            print(f"Inconsistency in {key}: {config1.get(key)} vs {config2.get(key)}")
            return False
    
    return True


if __name__ == "__main__":
    # Print configuration summary
    print("="*60)
    print("STANDARDIZED CONFIGURATION FOR THESIS RESULTS")
    print("="*60)
    
    baseline = get_standardized_model_config("baseline")
    fmsl = get_standardized_model_config("fmsl")
    
    print("\nBaseline Configuration:")
    for key, value in baseline.items():
        print(f"  {key}: {value}")
    
    print("\nFMSL Configuration (additional parameters):")
    for key, value in fmsl.items():
        if key.startswith('fmsl_'):
            print(f"  {key}: {value}")
    
    # Verify consistency
    print("\n" + "="*60)
    if verify_config_consistency(baseline, fmsl):
        print("Architecture consistency verified!")
    else:
        print("Architecture inconsistency detected!")
