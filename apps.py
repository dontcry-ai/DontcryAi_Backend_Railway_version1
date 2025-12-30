"""
Flask API for Baby Cry Classification (Validation Disabled)
With Auto-Shutdown Feature for Cost Optimization
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
import librosa
from datetime import datetime
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
print("✓ Model files ready!")

# ============================================================================
# AUTO-SHUTDOWN FOR IDLE TIMEOUT (Cost Saving)
# ============================================================================

from threading import Thread
import time
import signal

# Configuration: Shutdown after 10 minutes of inactivity
IDLE_SHUTDOWN_MINUTES = float(os.getenv("IDLE_SHUTDOWN_MINUTES", "10"))
last_activity = datetime.now()

def start_idle_monitor():
    """Monitor for inactivity and shutdown to save costs"""
    if IDLE_SHUTDOWN_MINUTES <= 0:
        print("✓ Auto-shutdown disabled (IDLE_SHUTDOWN_MINUTES=0)")
        return
    
    def check_idle():
        global last_activity
        check_interval = 60  # Check every 1 minute
        
        while True:
            time.sleep(check_interval)
            idle_seconds = (datetime.now() - last_activity).total_seconds()
            idle_minutes = idle_seconds / 60
            
            if idle_seconds > IDLE_SHUTDOWN_MINUTES * 60:
                print(f"\n{'='*70}")
                print(f"⏰ IDLE SHUTDOWN: No activity for {idle_minutes:.1f} minutes")
                print(f"💰 Shutting down to save costs...")
                print(f"🔄 Railway will auto-restart on next request")
                print(f"{'='*70}\n")
                os.kill(os.getpid(), signal.SIGTERM)
                break
            else:
                remaining = (IDLE_SHUTDOWN_MINUTES - idle_minutes)
                if remaining <= 5 or int(idle_minutes) % 5 == 0:  # Log at 5min intervals
                    print(f"⏱️  Idle: {idle_minutes:.1f}min | Shutdown in: {remaining:.1f}min")
    
    Thread(target=check_idle, daemon=True).start()
    print(f"✓ Auto-shutdown enabled: {IDLE_SHUTDOWN_MINUTES} minutes idle timeout")

# ============================================================================
# CRY CLASSIFIER (Validation removed)
# ============================================================================

class HuBERTClassifier(nn.Module):
    """5-class cry classifier"""
    
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
    """Cry type classifier"""
    
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
CRY_CLASSIFIER_MODEL_PATH = 'models/best_model.pth'
LABEL_ENCODER_PATH = 'models/label_encoder.json'

# Global model
cry_predictor = None


# Middleware to track activity on every request
@app.before_request
def track_activity():
    """Update last activity timestamp on each request"""
    global last_activity
    last_activity = datetime.now()


def init_models():
    """Initialize cry classifier"""
    global cry_predictor

    try:
        print("\nInitializing Cry Classifier...")
        cry_predictor = InfantCryPredictor(
            model_path=CRY_CLASSIFIER_MODEL_PATH,
            label_encoder_path=LABEL_ENCODER_PATH,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("✓ Cry classifier loaded successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error initializing cry classifier: {e}")
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

@app.route('/')
def home():
    """Root endpoint"""
    return jsonify({
        "message": "DontCryAI Backend API",
        "status": "running",
        "version": "2.0",
        "validation": "disabled",
        "auto_shutdown": f"{IDLE_SHUTDOWN_MINUTES} minutes idle timeout" if IDLE_SHUTDOWN_MINUTES > 0 else "disabled",
        "endpoints": {
            "health": "/api/health",
            "predict_upload": "/api/predict/upload",
            "predict_record": "/api/predict/record",
            "get_classes": "/api/classes"
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    idle_time = (datetime.now() - last_activity).total_seconds() / 60
    return jsonify({
        'status': 'healthy',
        'validation_enabled': False,
        'cry_predictor_loaded': cry_predictor is not None,
        'device': cry_predictor.device if cry_predictor else None,
        'classes': cry_predictor.classes if cry_predictor else None,
        'idle_minutes': round(idle_time, 2),
        'shutdown_in_minutes': round(IDLE_SHUTDOWN_MINUTES - idle_time, 2) if IDLE_SHUTDOWN_MINUTES > 0 else None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict/upload', methods=['POST'])
def predict_upload():
    """Predict from uploaded file"""
    try:
        if cry_predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not initialized',
                'message': 'The prediction model is not available. Please try again later.'
            }), 503
        
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file'
            }), 400
        
        confidence_threshold = float(request.form.get('confidence_threshold', 0.6))
        
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
                return jsonify({
                    'success': False,
                    'error': 'Audio conversion failed'
                }), 500
            os.remove(temp_path)
            temp_path = wav_path
        
        # Load audio
        audio, sr = load_audio_file(temp_path)
        
        # CLASSIFY (No validation)
        prediction_result = cry_predictor.predict_array(audio, sr, confidence_threshold)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'data': {
                'prediction': prediction_result,
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        print(f"Error in predict_upload: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict/record', methods=['POST'])
def predict_record():
    """Predict from recording"""
    try:
        if cry_predictor is None:
            return jsonify({
                'success': False,
                'error': 'Model not initialized',
                'message': 'The prediction model is not available. Please try again later.'
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        audio_format = data.get('format', 'base64')
        sample_rate = int(data.get('sample_rate', 16000))
        confidence_threshold = float(data.get('confidence_threshold', 0.6))
        
        # Parse audio
        if audio_format == 'base64':
            audio_base64 = data.get('audio_data')
            if not audio_base64:
                return jsonify({
                    'success': False,
                    'error': 'No audio_data provided'
                }), 400
            
            try:
                audio_bytes = base64.b64decode(audio_base64)
                temp_path = os.path.join(
                    app.config['UPLOAD_FOLDER'], 
                    f'temp_record_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav'
                )
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                
                audio, sr = load_audio_file(temp_path)
                os.remove(temp_path)
                
            except Exception as decode_error:
                print(f"Audio decode error: {decode_error}")
                return jsonify({
                    'success': False,
                    'error': 'Failed to decode audio data'
                }), 400
        
        elif audio_format == 'array':
            audio_array = data.get('audio_data')
            if not audio_array:
                return jsonify({
                    'success': False,
                    'error': 'No audio_data provided'
                }), 400
            audio = np.array(audio_array, dtype=np.float32)
            sr = sample_rate
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid format'
            }), 400
        
        # CLASSIFY (No validation)
        prediction_result = cry_predictor.predict_array(audio, sr, confidence_threshold)
        
        return jsonify({
            'success': True,
            'data': {
                'prediction': prediction_result,
                'timestamp': datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        print(f"Error in predict_record: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get cry types"""
    if cry_predictor:
        return jsonify({
            'success': True,
            'classes': cry_predictor.classes,
            'num_classes': cry_predictor.num_classes
        })
    return jsonify({
        'success': False,
        'error': 'Predictor not initialized'
    }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size: 10MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# INITIALIZE MODELS
# ============================================================================

print("=" * 70)
print("INITIALIZING BACKEND (VALIDATION DISABLED)")
print("=" * 70)

if not init_models():
    print("✗ Failed to initialize models.")
else:
    print(f"\n✓ Backend ready!")
    if cry_predictor:
        print(f"✓ Device: {cry_predictor.device}")
        print(f"✓ Cry classes: {cry_predictor.classes}")
    
    print("✓ Validation: DISABLED")
    
    # START IDLE MONITOR
    start_idle_monitor()
    
    print("=" * 70)

# For local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting local server on http://0.0.0.0:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
