#!/bin/bash

# Exit script on any error
set -e

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install Python 3 and pip
echo "Installing Python 3 and pip..."
sudo apt install -y python3-pip
pip install --upgrade pip


# Install required packages for flatbuffers
sudo apt-get remove --purge python3-flatbuffers -y
pip install --force-reinstall flatbuffers==23.5.26


# Install required Python packages
echo "Installing required Python packages..."
pip install ultralytics[export]

# Install PyTorch and torchvision for NVIDIA GPUs (arm64)
echo "Installing PyTorch and torchvision for NVIDIA GPUs..."
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

# Add NVIDIA CUDA repository
echo "Setting up NVIDIA CUDA repository..."
CUDA_KEYRING="cuda-keyring_1.1-1_all.deb"
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/${CUDA_KEYRING}
sudo dpkg -i ${CUDA_KEYRING}

# Install CUDA libraries
echo "Installing CUDA libraries..."
sudo apt-get install -y libcusparselt0 libcusparselt-dev

# Install ONNX Runtime GPU
echo "Installing ONNX Runtime GPU..."
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl

# (Optional) Install a specific ONNX Runtime GPU version (remove if unnecessary)
echo "Downloading and installing ONNX Runtime GPU v1.17.0..."
wget -q https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl

# Cleanup
echo "Cleaning up..."
rm -f ${CUDA_KEYRING}
rm -f onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl

# Reboot the system
echo "Setup complete! Rebooting the system..."
sudo reboot
