import os
import sys
import subprocess
import webbrowser
import time

def main():
    # Detect working directory
    if getattr(sys, 'frozen', False):
        app_dir = os.path.dirname(sys.executable)
        python_exec = os.path.join(app_dir, "python.exe")
        if not os.path.exists(python_exec):
            # fallback: use system python inside build_env
            python_exec = sys.executable
    else:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        python_exec = sys.executable

    os.chdir(app_dir)

    # Launch streamlit from this environment
    cmd = [
        python_exec,
        "-m",
        "streamlit",
        "run",
        "main.py",
        "--server.headless=true",
        "--server.port=8501"
    ]

    print("Launching Streamlit:", " ".join(cmd))

    process = subprocess.Popen(cmd, shell=False)

    # Wait for the app to start
    time.sleep(6)
    webbrowser.open("http://localhost:8501")

    # Keep alive
    process.wait()

if __name__ == "__main__":
    main()
