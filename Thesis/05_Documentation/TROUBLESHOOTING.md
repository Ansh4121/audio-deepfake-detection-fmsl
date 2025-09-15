# Troubleshooting Guide

## Common Issues and Solutions

### 1. Installation Issues

#### CUDA Not Available
**Error**: `CUDA not available`
**Solution**:
```bash
# Check CUDA installation
nvidia-smi

# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Import Errors
**Error**: `ModuleNotFoundError: No module named 'transformers'`
**Solution**:
```bash
# Install missing dependencies
pip install -r requirements.txt

# Or install specific package
pip install transformers
```

### 2. Training Issues

#### Out of Memory (OOM)
**Error**: `RuntimeError: CUDA out of memory`
**Solutions**:
- Reduce batch size: `--batch_size 16`
- Use gradient accumulation
- Enable mixed precision training
- Use CPU: `--device cpu`

#### Data Loading Issues
**Error**: `FileNotFoundError: No such file or directory`
**Solutions**:
- Check data paths in configuration
- Ensure ASVspoof2019 dataset is properly extracted
- Verify protocol file paths

#### Model Loading Issues
**Error**: `KeyError: 'model_state_dict'`
**Solutions**:
- Check model checkpoint format
- Use correct model class for loading
- Verify checkpoint compatibility

### 3. Evaluation Issues

#### Score File Errors
**Error**: `ValueError: could not convert string to float`
**Solutions**:
- Check score file format
- Verify protocol file format
- Ensure proper file encoding

#### Performance Issues
**Issue**: Slow evaluation
**Solutions**:
- Increase batch size
- Use GPU acceleration
- Enable mixed precision
- Use multiple workers

### 4. FMSL-Specific Issues

#### FMSL Configuration Errors
**Error**: `KeyError: 'fmsl_type'`
**Solutions**:
- Check FMSL configuration in model files
- Verify `fmsl_standardized_config.py` is imported
- Ensure proper FMSL parameters

#### Prototype Learning Issues
**Issue**: Poor FMSL performance
**Solutions**:
- Adjust prototype count: `fmsl_n_prototypes`
- Tune scale parameter: `fmsl_s`
- Adjust margin: `fmsl_m`
- Check data quality

### 5. System Issues

#### Permission Errors
**Error**: `PermissionError: [Errno 13] Permission denied`
**Solutions**:
- Check file permissions
- Run with appropriate privileges
- Ensure write access to directories

#### Path Issues
**Error**: `FileNotFoundError: [Errno 2] No such file or directory`
**Solutions**:
- Use absolute paths
- Check working directory
- Verify file existence

## Performance Optimization

### GPU Optimization
```bash
# Enable mixed precision
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Use optimized CUDA kernels
export TORCH_CUDNN_V8_API_ENABLED=1
```

### Memory Optimization
```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

# Use gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Debugging Tips

### Enable Debug Mode
```bash
# Set debug environment
export DEBUG=1
export LOG_LEVEL=DEBUG

# Run with verbose output
python -v your_script.py
```

### Check System Resources
```bash
# Check GPU memory
nvidia-smi

# Check CPU and RAM
htop

# Check disk space
df -h
```

## Getting Help

### Before Asking for Help
1. Check this troubleshooting guide
2. Search existing issues on GitHub
3. Check the logs for error messages
4. Verify your environment setup

### When Reporting Issues
Include:
- Error message and stack trace
- System information (OS, Python version, CUDA version)
- Steps to reproduce
- Configuration files used
- Log files

### Contact Information
- GitHub Issues: [Repository Issues](https://github.com/yourusername/audio-deepfake-detection-fmsl/issues)
- Email: your.email@university.edu
- Documentation: [Full Documentation](https://yourusername.github.io/audio-deepfake-detection-fmsl/)
