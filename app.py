"""
Part 4: Flask API with Baby Voice Validation
Integrates hybrid validator into existing Flask API (Phase 4)

This is your complete backend with validation.

Usage:
    python app_with_validator.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
import librosa
import soundfile as sf
from datetime import datetime
import io
import base64
from pydub import AudioSegment
import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2FeatureExtractor
import warnings
warnings.filterwarnings('ignore')
from download_models import download_models

# Download models before initializing
print("Checking for model files...")
download_models()

# ============================================================================
# BABY VOICE VALIDATOR CLASSES
# ============================================================================

class FrequencyAnalyzer:
    """Layer 1: Frequency-based validation"""
    
    def __init__(self):
        self.baby_f0_min = 300
        self.baby_f0_max = 600
        self.adult_male_max = 180
        self.adult_female_max = 255
    
    def extract_pitch(self, audio, sr=16000):
        """Extract fundamental frequency"""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) == 0:
                return None
            return np.median(valid_f0)
        except:
            return None
    
    def analyze(self, audio, sr=16000):
        """Analyze frequency characteristics"""
        median_pitch = self.extract_pitch(audio, sr)
        
        if median_pitch is None:
            return {
                'is_baby_likely': False,
                'confidence': 0.0,
                'median_pitch': None,
                'reasoning': 'No clear pitch detected'
            }
        
        if self.baby_f0_min <= median_pitch <= self.baby_f0_max:
            return {
                'is_baby_likely': True,
                'confidence': 0.8,
                'median_pitch': median_pitch,
                'reasoning': f'Pitch {median_pitch:.1f}Hz in baby range'
            }
        elif median_pitch <= self.adult_female_max:
            return {
                'is_baby_likely': False,
                'confidence': 0.7,
                'median_pitch': median_pitch,
                'reasoning': f'Pitch {median_pitch:.1f}Hz in adult range'
            }
        else:
            return {
                'is_baby_likely': False,
                'confidence': 0.6,
                'median_pitch': median_pitch,
                'reasoning': f'Pitch {median_pitch:.1f}Hz unusual (not baby)'
            }


class BabyVoiceCNN(nn.Module):
    """Layer 2: Neural classifier"""
    
    def __init__(self):
        super(BabyVoiceCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class HybridBabyVoiceValidator:
    """Hybrid validator combining both layers"""
    
    def __init__(self, neural_model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Layer 1
        self.freq_analyzer = FrequencyAnalyzer()
        
        # Layer 2
        self.neural_model = BabyVoiceCNN()
        checkpoint = torch.load(neural_model_path, map_location=self.device, weights_only=True)
        self.neural_model.load_state_dict(checkpoint['model_state_dict'])
        self.neural_model.to(self.device)
        self.neural_model.eval()
    
    def validate_audio_array(self, audio, sr=16000, freq_weight=0.3, neural_weight=0.7, threshold=0.65):
        """
        Validate if audio is baby voice
        
        Returns:
            dict: {
                'is_baby_voice': bool,
                'combined_confidence': float,
                'frequency_analysis': dict,
                'neural_classification': dict
            }
        """
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Layer 1: Frequency Analysis
        freq_result = self.freq_analyzer.analyze(audio, sr)
        freq_score = freq_result['confidence'] if freq_result['is_baby_likely'] else (1 - freq_result['confidence'])
        
        # Layer 2: Neural Classification
        try:
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
            
            # Pad or truncate to consistent size
            target_width = 313  # Standard for 5-second audio
            if mel_spec_db.shape[1] < target_width:
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_width - mel_spec_db.shape[1])))
            else:
                mel_spec_db = mel_spec_db[:, :target_width]
            
            features = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.neural_model(features)
                probs = torch.softmax(outputs, dim=1)
                neural_confidence = probs[0][1].item()
        except Exception as e:
            print(f"Neural classification error: {e}")
            neural_confidence = 0.5
        
        # Combine scores
        combined_confidence = (freq_weight * freq_score) + (neural_weight * neural_confidence)
        is_baby = combined_confidence >= threshold
        
        return {
            'is_baby_voice': is_baby,
            'combined_confidence': float(combined_confidence),
            'frequency_analysis': {
                'score': float(freq_score),
                'median_pitch': float(freq_result['median_pitch']) if freq_result['median_pitch'] else None,
                'reasoning': freq_result['reasoning']
            },
            'neural_classification': {
                'baby_probability': float(neural_confidence),
                'non_baby_probability': float(1 - neural_confidence)
            },
            'threshold': threshold
        }


# ============================================================================
# CRY CLASSIFIER (YOUR EXISTING PHASE 2 MODEL)
# ============================================================================

class HuBERTClassifier(nn.Module):
    """Your existing 5-class cry classifier"""
    
    def __init__(self, num_classes, hubert_model_name="facebook/hubert-base-ls960", freeze_encoder=False):
        super(HuBERTClassifier, self).__init__()
        self.hubert = HubertModel.from_pretrained(hubert_model_name)
        
        if freeze_encoder:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        hidden_size = self.hubert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_values):
        outputs = self.hubert(input_values)
        hidden_states = outputs.last_hidden_state
        pooled = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled)
        return logits


class AudioPreprocessor:
    """Preprocess audio"""
    
    def __init__(self, target_sr=16000, target_duration=5.0):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_length = int(target_sr * target_duration)
    
    def preprocess_audio_array(self, audio, sr=None):
        """Preprocess audio array"""
        if sr is not None and sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
        
        if len(audio_trimmed) > self.target_length:
            audio_trimmed = audio_trimmed[:self.target_length]
        elif len(audio_trimmed) < self.target_length:
            padding = self.target_length - len(audio_trimmed)
            audio_trimmed = np.pad(audio_trimmed, (0, padding), mode='constant')
        
        return audio_trimmed


class InfantCryPredictor:
    """Your existing cry type classifier"""
    
    def __init__(self, model_path, label_encoder_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        with open(label_encoder_path, 'r') as f:
            label_data = json.load(f)
        self.classes = label_data['classes']
        self.num_classes = len(self.classes)
        
        self.model = HuBERTClassifier(
            num_classes=self.num_classes,
            hubert_model_name="facebook/hubert-base-ls960",
            freeze_encoder=True
        )
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    
    def predict_array(self, audio_array, sample_rate=16000, confidence_threshold=0.5):
        """Predict cry type"""
        audio = self.preprocessor.preprocess_audio_array(audio_array, sr=sample_rate)
        
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_values)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = self.classes[predicted_idx.item()]
            all_probs = {self.classes[i]: float(probabilities[0][i].item()) 
                        for i in range(self.num_classes)}
        
        result = {
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'all_probabilities': all_probs,
            'meets_threshold': confidence >= confidence_threshold,
            'audio_duration': len(audio) / 16000
        }
        
        return result


# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model paths
VALIDATOR_MODEL_PATH = 'validator_models/neural_classifier.pth'
CRY_CLASSIFIER_MODEL_PATH = 'models/best_model.pth'
LABEL_ENCODER_PATH = 'models/label_encoder.json'

# Global models
validator = None
cry_predictor = None


def init_models():
    """Initialize both models"""
    global validator, cry_predictor
    
    try:
        print("\nInitializing Baby Voice Validator...")
        validator = HybridBabyVoiceValidator(
            neural_model_path=VALIDATOR_MODEL_PATH,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✓ Validator loaded successfully!")
        
        print("\nInitializing Cry Classifier...")
        cry_predictor = InfantCryPredictor(
            model_path=CRY_CLASSIFIER_MODEL_PATH,
            label_encoder_path=LABEL_ENCODER_PATH,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✓ Cry classifier loaded successfully!")
        
        return True
    except Exception as e:
        print(f"✗ Error initializing models: {e}")
        return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def convert_to_wav(input_path, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio.export(output_path, format='wav')
        return True
    except Exception as e:
        print(f"Conversion error: {e}")
        return False


def load_audio_file(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        return audio, sr
    except Exception as e:
        raise Exception(f"Error loading audio: {e}")


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'validator_loaded': validator is not None,
        'cry_predictor_loaded': cry_predictor is not None,
        'device': cry_predictor.device if cry_predictor else None,
        'classes': cry_predictor.classes if cry_predictor else None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/validate/audio', methods=['POST'])
def validate_audio_only():
    """
    NEW ENDPOINT: Validate if audio is baby voice (no classification)
    
    Request:
        - file: Audio file
        - threshold: Optional validation threshold (default 0.65)
    
    Response:
        - is_baby_voice: bool
        - combined_confidence: float
        - frequency_analysis: dict
        - neural_classification: dict
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        threshold = float(request.form.get('threshold', 0.65))
        
        # Save and process file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)
        
        # Convert if needed
        if not filename.lower().endswith('.wav'):
            wav_path = temp_path.rsplit('.', 1)[0] + '.wav'
            if not convert_to_wav(temp_path, wav_path):
                os.remove(temp_path)
                return jsonify({'error': 'Audio conversion failed'}), 500
            os.remove(temp_path)
            temp_path = wav_path
        
        # Load audio
        audio, sr = load_audio_file(temp_path)
        
        # Validate
        validation_result = validator.validate_audio_array(audio, sr, threshold=threshold)
        
        # Cleanup
        os.remove(temp_path)
        
        validation_result['timestamp'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'data': validation_result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    """
    MODIFIED: Predict with validation
    
    Request:
        - file: Audio file
        - confidence_threshold: Optional (default 0.6)
        - validation_threshold: Optional (default 0.65)
    
    Response:
        - validation: Validation result
        - prediction: Cry type classification (only if validated)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
        validation_threshold = float(request.form.get('validation_threshold', 0.65))
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{filename}"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)
        
        # Convert if needed
        if not filename.lower().endswith('.wav'):
            wav_path = temp_path.rsplit('.', 1)[0] + '.wav'
            if not convert_to_wav(temp_path, wav_path):
                os.remove(temp_path)
                return jsonify({'error': 'Audio conversion failed'}), 500
            os.remove(temp_path)
            temp_path = wav_path
        
        # Load audio
        audio, sr = load_audio_file(temp_path)
        
        # STEP 1: VALIDATE (Baby voice check)
        validation_result = validator.validate_audio_array(audio, sr, threshold=validation_threshold)
        
        # If NOT baby voice, return early
        if not validation_result['is_baby_voice']:
            os.remove(temp_path)
            return jsonify({
                'success': False,
                'error': 'Audio validation failed',
                'message': 'This does not appear to be a baby cry. Please upload baby cry audio only.',
                'validation': validation_result,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # STEP 2: CLASSIFY (Cry type prediction)
        prediction_result = cry_predictor.predict_array(audio, sr, confidence_threshold)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'data': {
                'validation': validation_result,
                'prediction': prediction_result,
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/record', methods=['POST'])
def predict_record():
    """
    MODIFIED: Predict from recording with validation
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        audio_format = data.get('format', 'base64')
        sample_rate = int(data.get('sample_rate', 16000))
        confidence_threshold = float(data.get('confidence_threshold', 0.6))
        validation_threshold = float(data.get('validation_threshold', 0.65))
        
        # Parse audio
        if audio_format == 'base64':
            audio_base64 = data.get('audio_data')
            if not audio_base64:
                return jsonify({'error': 'No audio_data provided'}), 400
            
            audio_bytes = base64.b64decode(audio_base64)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_record_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav')
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            audio, sr = load_audio_file(temp_path)
            os.remove(temp_path)
        
        elif audio_format == 'array':
            audio_array = data.get('audio_data')
            if not audio_array:
                return jsonify({'error': 'No audio_data provided'}), 400
            audio = np.array(audio_array, dtype=np.float32)
            sr = sample_rate
        else:
            return jsonify({'error': 'Invalid format'}), 400
        
        # STEP 1: VALIDATE
        validation_result = validator.validate_audio_array(audio, sr, threshold=validation_threshold)
        
        if not validation_result['is_baby_voice']:
            return jsonify({
                'success': False,
                'error': 'Audio validation failed',
                'message': 'This does not appear to be a baby cry.',
                'validation': validation_result,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # STEP 2: CLASSIFY
        prediction_result = cry_predictor.predict_array(audio, sr, confidence_threshold)
        
        return jsonify({
            'success': True,
            'data': {
                'validation': validation_result,
                'prediction': prediction_result,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get cry types"""
    if cry_predictor:
        return jsonify({
            'classes': cry_predictor.classes,
            'num_classes': cry_predictor.num_classes
        })
    return jsonify({'error': 'Predictor not initialized'}), 500


@app.route('/api/validator/status', methods=['GET'])
def validator_status():
    """NEW: Get validator status"""
    if validator:
        return jsonify({
            'status': 'active',
            'model_type': 'Hybrid (Frequency + Neural)',
            'validation_threshold': 0.65,
            'weights': {
                'frequency': 0.3,
                'neural': 0.7
            }
        })
    return jsonify({'error': 'Validator not initialized'}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size: 10MB'}), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':

    port = int(os.environ.get('PORT', 5000))

    print("=" * 70)
    print("FLASK BACKEND WITH BABY VOICE VALIDATION")
    print("=" * 70)
    
    if not init_models():
        print("✗ Failed to initialize models. Exiting.")
        exit(1)
    
    print(f"\n✓ Backend ready with validation!")
    print(f"✓ Device: {cry_predictor.device}")
    print(f"✓ Cry classes: {cry_predictor.classes}")
    
    print("\n" + "=" * 70)
    print("API ENDPOINTS")
    print("=" * 70)
    print("GET  /api/health              - Health check")
    print("GET  /api/validator/status    - Validator status (NEW)")
    print("POST /api/validate/audio      - Validate baby voice only (NEW)")
    print("POST /api/predict/upload      - Upload + Validate + Predict")
    print("POST /api/predict/record      - Record + Validate + Predict")
    print("GET  /api/classes             - Get cry types")
    print("=" * 70)
    
    print("\n🚀 Starting server on http://localhost:5000")
    print("=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)