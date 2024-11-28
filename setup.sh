#!/bin/bash

# Get the current directory
path=$(pwd)
sudo apt install python3.12-env -y
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