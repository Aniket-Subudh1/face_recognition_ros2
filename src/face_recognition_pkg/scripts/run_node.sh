#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

# Activate virtual environment
source "${WORKSPACE_DIR}/.venv/bin/activate"

# Add the virtual environment's site-packages to PYTHONPATH
SITE_PACKAGES="${WORKSPACE_DIR}/.venv/lib/python3.12/site-packages"
export PYTHONPATH="${SITE_PACKAGES}:${PYTHONPATH}"

# Run the actual node
python3 "${WORKSPACE_DIR}/src/face_recognition_pkg/face_recognition_pkg/face_recognition_node.py" "$@"
