import os
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit application."""
    # Determine the directory of this script
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Check if .env file exists
    env_file = script_dir / ".env"
    if not env_file.exists():
        # If .env.example exists, copy it to .env
        env_example = script_dir / ".env.example"
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("Created .env file from .env.example. Please edit it to add your API keys.")
        else:
            print("Warning: No .env file found. Please create one with your API keys.")
    
    # Run the Streamlit app
    app_path = script_dir / "app.py"
    try:
        # Execute streamlit run command
        cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", "8501"]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping the application...")
    except Exception as e:
        print(f"Error running the application: {e}")

if __name__ == "__main__":
    main() 