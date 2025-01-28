#!/bin/bash

set -e  # Exit on error

echo "Setting up ROS2 face recognition workspace..."

# Deactivate any active virtual environment
deactivate 2>/dev/null || true

# Remove old virtual environment
rm -rf .venv

# Create new virtual environment with system packages
python3 -m venv .venv --system-site-packages

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Clean old build files
rm -rf build/ install/ log/

# Set PYTHONPATH to include workspace
export PYTHONPATH=$PWD/src:$PYTHONPATH

# Build the package
colcon build --packages-select face_recognition_pkg --symlink-install

# Source the setup
source install/setup.bash

echo "Setup complete! Environment is ready."
