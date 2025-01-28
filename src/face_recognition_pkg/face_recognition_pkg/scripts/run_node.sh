#!/usr/bin/env bash
# scripts/run_node.sh

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
source "${WORKSPACE_DIR}/.venv/bin/activate"

# Add the virtual environment's site-packages to PYTHONPATH
SITE_PACKAGES_DIR="${WORKSPACE_DIR}/.venv/lib/python3.12/site-packages"
export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH}"

# Run the actual node with all arguments passed to this script
python3 "${WORKSPACE_DIR}/src/face_recognition_pkg/face_recognition_pkg/face_recognition_node.py" "$@"