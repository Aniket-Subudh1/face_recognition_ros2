# src/face_recognition_pkg/face_recognition_pkg/scripts/run_node.py

import os
import sys
import subprocess

def main():
    # Get the correct workspace directory
    script_path = os.path.abspath(__file__)
    face_recognition_pkg_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    workspace_dir = os.path.dirname(os.path.dirname(face_recognition_pkg_dir))
    
    # Debug prints
    print(f"Script path: {script_path}")
    print(f"Face recognition package dir: {face_recognition_pkg_dir}")
    print(f"Workspace directory: {workspace_dir}")
    
    # Define correct paths
    venv_python = os.path.join(workspace_dir, '.venv', 'bin', 'python3')
    node_script = os.path.join(face_recognition_pkg_dir, 'face_recognition_pkg', 'face_recognition_node.py')
    
    print(f"Virtual env Python: {venv_python}")
    print(f"Node script: {node_script}")
    
    # Verify paths exist
    if not os.path.exists(venv_python):
        print(f"Error: Virtual environment Python not found at {venv_python}")
        print("\nChecking workspace structure:")
        os.system(f"ls -la {workspace_dir}")
        sys.exit(1)
        
    if not os.path.exists(node_script):
        print(f"Error: Node script not found at {node_script}")
        print("\nChecking package structure:")
        os.system(f"ls -la {os.path.dirname(node_script)}")
        sys.exit(1)
    
    # Set up environment
    env = os.environ.copy()
    site_packages = os.path.join(workspace_dir, '.venv', 'lib', 'python3.10', 'site-packages')
    env['PYTHONPATH'] = f"{site_packages}:{workspace_dir}/src:{env.get('PYTHONPATH', '')}"
    
    try:
        subprocess.run([venv_python, node_script] + sys.argv[1:], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running node: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()