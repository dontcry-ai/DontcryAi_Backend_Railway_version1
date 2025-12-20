import os
import gdown

def download_models():
    """Download models from Google Drive on startup"""
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('validator_models', exist_ok=True)
    
    # Model files with Google Drive File IDs
    models = {
        'validator_models/neural_classifier.pth': '1uoRlHZl1xy_DRnIAPbWv7Aa0Cm_i4ycI',
        'models/best_model.pth': '1ukJZvHxYqWLiIEAJ4RN6f6XqHE0c81di',
        'models/label_encoder.json': '18-PHK9vI4QTmi2vUfFd6Zlxh0a3jWTE0'
    }
    
    for path, file_id in models.items():
        if not os.path.exists(path):
            print(f"📥 Downloading {path}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, path, quiet=False)
                print(f"✓ Downloaded {path}")
            except Exception as e:
                print(f"✗ Failed to download {path}: {e}")
                raise
        else:
            print(f"✓ {path} already exists")

if __name__ == '__main__':
    download_models()