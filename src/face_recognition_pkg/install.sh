#!/bin/bash
# install.sh

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install numpy<2 first to avoid compatibility issues
pip install 'numpy<2'

# Install other required packages
pip install torch torchvision opencv-python face-recognition rclpy pyyaml

# Clean the workspace
rm -rf build/ install/ log/

# Build the package
colcon build --packages-select face_recognition_pkg --symlink-install

# Source the setup files
source install/setup.bash

echo "Installation complete!"
