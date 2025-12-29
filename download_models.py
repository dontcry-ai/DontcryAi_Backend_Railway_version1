import os
import requests
from tqdm import tqdm

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive using direct download link"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, "wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination) as pbar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


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
            try:
                download_file_from_google_drive(file_id, path)
                print(f"✓ Downloaded {path}")
            except Exception as e:
                print(f"✗ Failed to download {path}: {e}")
                raise
        else:
            print(f"✓ {path} already exists")


if __name__ == '__main__':
    download_models()
