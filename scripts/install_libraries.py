import subprocess

# List of required libraries
required_libraries = [
    "numpy",
    "opencv-python",
    "tqdm",
    "scikit-learn",
    "matplotlib",
    "tensorflow"
]

# Check and install required libraries
for lib in required_libraries:
    try:
        __import__(lib)
        print(f"{lib} is already installed")
    except ImportError:
        print(f"Installing {lib}...")
        subprocess.check_call(["pip", "install", lib])

print("All required libraries are installed.")
