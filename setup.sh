#!/bin/bash

# Update and upgrade system packages
sudo apt update && sudo apt upgrade -y

# Install pip for Python package management
sudo apt install -y python3-pip

# Install the NVIDIA driver (replace 550 with the correct version for your instance)
sudo apt-get install -y nvidia-driver-550

# Install the CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit

# Get the current directory
path=$(pwd)

# Create a Python virtual environment in the current directory
python3 -m venv $path/venv

# Activate the virtual environment
source $path/venv/bin/activate

# Install required Python packages
pip install --upgrade pip  # Ensure pip is up-to-date
pip install -r requirements.txt

# Install FastAI version 2.7.18 explicitly
pip install fastai==2.7.18


python3 $path/save_pwd.py
echo "Setup complete! Virtual environment activated at $path/venv and PWD saved."

