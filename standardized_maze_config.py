#!/usr/bin/env python3
"""
Standardized configuration for all maze models
This ensures architectural consistency across all models
"""

# Standardized configuration for fair comparison
STANDARDIZED_CONFIG = {
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
    'fmsl': {
        'fmsl_type': 'prototype',
        'fmsl_n_prototypes': 3,
        'fmsl_s': 32.0,
        'fmsl_m': 0.45,
        'fmsl_enable_lsa': False
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

def get_standardized_config(model_type="baseline"):
    """
    Get standardized configuration for maze models
    
    Args:
        model_type: "baseline" or "fmsl"
    
    Returns:
        dict: Complete model configuration
    """
    config = {}
    
    # Add architecture config
    config.update(STANDARDIZED_CONFIG['architecture'])
    
    # Add Wav2Vec2 config
    config.update(STANDARDIZED_CONFIG['wav2vec2'])
    
    # Add training config
    config.update(STANDARDIZED_CONFIG['training'])
    
    # Add FMSL config if needed
    if model_type == "fmsl":
        config.update(STANDARDIZED_CONFIG['fmsl'])
    
    return config

def verify_config_consistency(config1, config2, keys_to_check=None):
    """
    Verify configuration consistency between two models
    
    Args:
        config1: First configuration dict
        config2: Second configuration dict
        keys_to_check: List of keys to check (if None, checks all)
    
    Returns:
        bool: True if consistent, False otherwise
    """
    if keys_to_check is None:
        keys_to_check = STANDARDIZED_CONFIG['architecture'].keys()
    
    for key in keys_to_check:
        if config1.get(key) != config2.get(key):
            print(f"Inconsistency found in {key}: {config1.get(key)} vs {config2.get(key)}")
            return False
    
    return True

if __name__ == "__main__":
    # Example usage
    baseline_config = get_standardized_config("baseline")
    fmsl_config = get_standardized_config("fmsl")
    
    print("Baseline configuration:")
    for key, value in baseline_config.items():
        print(f"  {key}: {value}")
    
    print("\nFMSL configuration:")
    for key, value in fmsl_config.items():
        print(f"  {key}: {value}")
