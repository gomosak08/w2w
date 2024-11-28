#!/bin/bash

# Update and upgrade system packages
sudo apt update && sudo apt upgrade -y

# Install pip for Python package management
sudo apt install -y python3-pip

# Install the NVIDIA driver (replace 550 with the correct version for your instance)
sudo apt-get install -y nvidia-driver-550

# Install the CUDA toolkit
sudo apt install -y nvidia-cuda-toolkit
####
sudo reboot

