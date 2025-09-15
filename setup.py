#!/usr/bin/env python3
"""
Setup script for Audio Deepfake Detection using FMSL
"""

from setuptools import setup, find_packages
import os

# Read README
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="audio-deepfake-detection-fmsl",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="Audio Deepfake Detection using Frequency-Modulated Spectral Loss (FMSL)",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio-deepfake-detection-fmsl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fmsl-train=Thesis.01_Models.01_Baseline_Models.main:main",
            "fmsl-eval=Thesis.02_Evaluation_Scripts.comprehensive_evaluation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "Thesis": [
            "07_Configuration_Files/*.yaml",
            "08_Notebooks/*.ipynb",
        ],
    },
)
