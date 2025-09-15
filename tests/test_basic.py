#!/usr/bin/env python3
"""
Basic tests for Audio Deepfake Detection using FMSL
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_pytorch_installation():
    """Test PyTorch installation and basic functionality"""
    assert torch.__version__ is not None
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Test basic tensor operations
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    z = torch.mm(x, y)
    assert z.shape == (2, 4)
    print("✅ PyTorch tensor operations working")

def test_cuda_availability():
    """Test CUDA availability"""
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        
        # Test basic CUDA operations
        x = torch.randn(2, 3).cuda()
        y = torch.randn(3, 4).cuda()
        z = torch.mm(x, y)
        assert z.is_cuda
        print("✅ CUDA operations working")
    else:
        print("⚠️ CUDA not available, using CPU")

def test_imports():
    """Test essential imports"""
    try:
        import librosa
        print(f"✅ Librosa version: {librosa.__version__}")
    except ImportError:
        pytest.fail("Librosa not installed")
    
    try:
        import soundfile
        print(f"✅ SoundFile version: {soundfile.__version__}")
    except ImportError:
        pytest.fail("SoundFile not installed")
    
    try:
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
    except ImportError:
        pytest.fail("Transformers not installed")
    
    try:
        import sklearn
        print(f"✅ Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        pytest.fail("Scikit-learn not installed")

def test_model_imports():
    """Test model imports"""
    try:
        # Test importing standardized config
        from Thesis.standardized_maze_config import get_standardized_config
        config = get_standardized_config("baseline")
        assert config is not None
        print("✅ Standardized config import working")
    except ImportError as e:
        pytest.fail(f"Failed to import standardized config: {e}")

def test_audio_processing():
    """Test basic audio processing"""
    import librosa
    import soundfile as sf
    
    # Create test audio
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Test librosa processing
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    assert mfcc.shape[0] == 13
    print("✅ Librosa MFCC extraction working")
    
    # Test soundfile
    test_file = "test_audio.wav"
    sf.write(test_file, audio, sample_rate)
    
    loaded_audio, loaded_sr = sf.read(test_file)
    assert np.allclose(audio, loaded_audio)
    assert loaded_sr == sample_rate
    print("✅ SoundFile read/write working")
    
    # Cleanup
    os.remove(test_file)

def test_model_configuration():
    """Test model configuration loading"""
    try:
        from Thesis.standardized_maze_config import get_standardized_config
        
        # Test baseline config
        baseline_config = get_standardized_config("baseline")
        assert "filts" in baseline_config
        assert "nb_fc_node" in baseline_config
        assert "nb_classes" in baseline_config
        print("✅ Baseline configuration loading working")
        
        # Test FMSL config
        fmsl_config = get_standardized_config("fmsl")
        assert "fmsl_type" in fmsl_config
        assert "fmsl_n_prototypes" in fmsl_config
        print("✅ FMSL configuration loading working")
        
    except Exception as e:
        pytest.fail(f"Configuration loading failed: {e}")

if __name__ == "__main__":
    # Run tests
    test_pytorch_installation()
    test_cuda_availability()
    test_imports()
    test_model_imports()
    test_audio_processing()
    test_model_configuration()
    print("\n🎉 All basic tests passed!")
