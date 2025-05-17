import subprocess
import sys

def setup_environment():
    """Set up the environment with correct package versions."""
    print("Setting up environment...")
    
    # Uninstall potentially conflicting packages
    packages_to_uninstall = [
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "jax",
        "jaxlib"
    ]
    
    for package in packages_to_uninstall:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package])
    
    # Install packages in the correct order
    packages = [
        "numpy==1.23.5",
        "torch==2.1.0",
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "xgboost==1.7.6",
        "gradio==4.19.2",
        "shap==0.42.1",
        "matplotlib==3.7.2",
        "seaborn==0.12.2",
        "plotly==5.15.0",
        "nltk==3.8.1",
        "spacy==3.7.2",
        "tqdm==4.66.1",
        "joblib==1.3.2"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    print("Environment setup complete!")

if __name__ == "__main__":
    setup_environment() 