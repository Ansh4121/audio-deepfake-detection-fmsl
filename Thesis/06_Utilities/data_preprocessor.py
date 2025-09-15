#!/usr/bin/env python3
"""
Data preprocessing utilities for ASVspoof2019 LA dataset
"""

import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def preprocess_audio(file_path, target_sr=16000, duration=None):
    """
    Preprocess audio file for training
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        duration: Target duration in seconds (None for no trimming)
    
    Returns:
        audio: Preprocessed audio array
        sr: Sample rate
    """
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=target_sr)
        
        # Trim silence
        audio, _ = librosa.effects.trim(audio)
        
        # Pad or truncate to target duration
        if duration is not None:
            target_length = int(duration * target_sr)
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
        
        return audio, sr
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def create_dataset_manifest(data_dir, output_file, protocol_file=None):
    """
    Create dataset manifest for training
    
    Args:
        data_dir: Directory containing audio files
        output_file: Output manifest file path
        protocol_file: Protocol file path (optional)
    """
    manifest_data = []
    
    # Get all audio files
    audio_files = []
    for ext in ['*.flac', '*.wav', '*.mp3']:
        audio_files.extend(Path(data_dir).rglob(ext))
    
    print(f"Found {len(audio_files)} audio files")
    
    for file_path in tqdm(audio_files, desc="Processing files"):
        # Get relative path
        rel_path = file_path.relative_to(data_dir)
        
        # Determine label from filename or protocol
        label = "unknown"
        if protocol_file and os.path.exists(protocol_file):
            # Read protocol file to get labels
            with open(protocol_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[1] in str(rel_path):
                        label = parts[4]  # bonafide or spoof
                        break
        
        # Get file info
        try:
            audio, sr = preprocess_audio(str(file_path))
            if audio is not None:
                manifest_data.append({
                    'file_path': str(rel_path),
                    'label': label,
                    'duration': len(audio) / sr,
                    'sample_rate': sr
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save manifest
    df = pd.DataFrame(manifest_data)
    df.to_csv(output_file, index=False)
    print(f"Created manifest with {len(df)} files")
    print(f"Label distribution:\n{df['label'].value_counts()}")

def validate_dataset(data_dir, protocol_file):
    """
    Validate dataset integrity
    
    Args:
        data_dir: Directory containing audio files
        protocol_file: Protocol file path
    """
    print("Validating dataset...")
    
    # Check protocol file
    if not os.path.exists(protocol_file):
        print(f"❌ Protocol file not found: {protocol_file}")
        return False
    
    # Read protocol file
    protocol_data = []
    with open(protocol_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                protocol_data.append({
                    'file_id': parts[1],
                    'label': parts[4]
                })
    
    print(f"Protocol file contains {len(protocol_data)} entries")
    
    # Check audio files
    missing_files = []
    for entry in protocol_data:
        file_id = entry['file_id']
        # Look for file with this ID
        found = False
        for ext in ['flac', 'wav', 'mp3']:
            file_path = os.path.join(data_dir, f"{file_id}.{ext}")
            if os.path.exists(file_path):
                found = True
                break
        if not found:
            missing_files.append(file_id)
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} files")
        print("First 10 missing files:", missing_files[:10])
        return False
    else:
        print("✅ All files found")
        return True

def main():
    parser = argparse.ArgumentParser(description='Data preprocessing utilities')
    parser.add_argument('--data_dir', required=True, help='Data directory')
    parser.add_argument('--protocol_file', help='Protocol file path')
    parser.add_argument('--output_manifest', help='Output manifest file')
    parser.add_argument('--validate', action='store_true', help='Validate dataset')
    parser.add_argument('--create_manifest', action='store_true', help='Create manifest')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_dataset(args.data_dir, args.protocol_file)
    
    if args.create_manifest and args.output_manifest:
        create_dataset_manifest(args.data_dir, args.output_manifest, args.protocol_file)

if __name__ == "__main__":
    main()
