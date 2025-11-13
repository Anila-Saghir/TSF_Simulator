import subprocess
import sys
import os

# Get the folder of this script
folder = os.path.dirname(os.path.abspath(__file__))

# Run Streamlit pointing to main.py
subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.join(folder, "main.py")])
