"""
Setup script to initialize the project
Run: python setup.py
"""

import os
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    directories = ['data', 'models', 'src']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}/")

def install_requirements():
    """Install required packages"""
    print("\nInstalling required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing packages. Please install manually:")
        print("   pip install -r requirements.txt")

def train_model():
    """Train the model"""
    print("\nTraining model...")
    try:
        subprocess.check_call([sys.executable, "src/train_model.py"])
        print("✓ Model trained successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error training model. Please train manually:")
        print("   python src/train_model.py")

def main():
    print("="*70)
    print("DIABETES PREDICTION SYSTEM - SETUP")
    print("="*70)
    
    # Create directories
    print("\n[1/3] Creating project directories...")
    create_directories()
    
    # Install requirements
    print("\n[2/3] Installing dependencies...")
    response = input("Install required packages? (y/n): ").lower()
    if response == 'y':
        install_requirements()
    else:
        print("Skipped. Install manually with: pip install -r requirements.txt")
    
    # Train model
    print("\n[3/3] Training model...")
    response = input("Train the model now? (y/n): ").lower()
    if response == 'y':
        train_model()
    else:
        print("Skipped. Train manually with: python src/train_model.py")
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. If you haven't already, train the model:")
    print("     python src/train_model.py")
    print("\n  2. Run the web application:")
    print("     streamlit run app.py")
    print("\n  3. (Optional) Run the REST API:")
    print("     python api.py")
    print("="*70)

if __name__ == "__main__":
    main()