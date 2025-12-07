"""
Flask web application for fall detection.

This app provides a web interface for uploading videos and detecting falls
using various trained models.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
import numpy as np

# Add parent directory to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fall_detector_main import FallDetectorMain


# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
RESULTS_FOLDER = Path(__file__).parent / 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

# Ensure folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['RESULTS_FOLDER'] = str(RESULTS_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'fall-detection-secret-key-change-in-production'

# Model configurations
MODEL_CONFIGS = {
    'ensemble': {
        'name': 'Ensemble Detector',
        'description': 'Combines physics, temporal, and geometry detectors (no neural network needed)',
        'detector_type': 'ensemble',
        'model_path': None,
        'requires_model': False,
    },
    'kth_neural': {
        'name': 'KTH Neural (Improved Hybrid)',
        'description': 'LSTM-based hybrid model trained on KTH dataset',
        'detector_type': 'neural',
        'model_path': PROJECT_ROOT / 'models' / 'improved_hybrid_detector.pt',
        'requires_model': True,
    },
    'kth_basic': {
        'name': 'KTH Basic (Hybrid)',
        'description': 'Basic hybrid model trained on KTH dataset',
        'detector_type': 'neural',
        'model_path': PROJECT_ROOT / 'models' / 'checkpoints' / 'hybrid_fall_detector.pt',
        'requires_model': True,
    },
    'kth_finetuned': {
        'name': 'KTH Fine-tuned',
        'description': 'Autoencoder fine-tuned on Kaggle dataset',
        'detector_type': 'neural',
        'model_path': PROJECT_ROOT / 'models' / 'checkpoints' / 'pose_autoencoder_kaggle_finetuned.pt',
        'requires_model': True,
    },
    'physics': {
        'name': 'Physics-Based Detector',
        'description': 'Pure physics-based rules (high recall)',
        'detector_type': 'physics',
        'model_path': None,
        'requires_model': False,
    },
    'temporal': {
        'name': 'Temporal Pattern Detector',
        'description': 'Temporal motion pattern analysis',
        'detector_type': 'temporal',
        'model_path': None,
        'requires_model': False,
    },
    'geometry': {
        'name': 'Pose Geometry Detector',
        'description': 'Body geometry and pose-based detection',
        'detector_type': 'geometry',
        'model_path': None,
        'requires_model': False,
    },
}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_available_models():
    """Get list of available models (only those with existing model files)."""
    available = []
    for model_id, config in MODEL_CONFIGS.items():
        if not config['requires_model'] or (config['model_path'] and config['model_path'].exists()):
            available.append({
                'id': model_id,
                'name': config['name'],
                'description': config['description'],
            })
    return available


@app.route('/')
def index():
    """Render main page."""
    available_models = get_available_models()
    return render_template('index.html', models=available_models)


@app.route('/api/models')
def api_models():
    """API endpoint to get available models."""
    return jsonify(get_available_models())


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle video upload."""
    # Check if file is present
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save file with secure filename
        original_filename = file.filename or 'video.mp4'
        filename = secure_filename(original_filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = UPLOAD_FOLDER / unique_filename
        
        file.save(str(filepath))
        
        return jsonify({
            'success': True,
            'filename': unique_filename,
            'original_filename': filename,
            'size': os.path.getsize(filepath),
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to upload file: {str(e)}'}), 500


@app.route('/api/detect', methods=['POST'])
def detect_fall():
    """Process video and detect falls."""
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filename = data.get('filename')
        model_id = data.get('model', 'ensemble')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        if model_id not in MODEL_CONFIGS:
            return jsonify({'error': f'Invalid model: {model_id}'}), 400
        
        # Check if file exists
        video_path = UPLOAD_FOLDER / filename
        if not video_path.exists():
            return jsonify({'error': 'Video file not found'}), 404
        
        # Get model configuration
        model_config = MODEL_CONFIGS[model_id]
        
        # Check if model file exists (if required)
        if model_config['requires_model'] and not model_config['model_path'].exists():
            return jsonify({'error': f'Model file not found: {model_config["model_path"]}'}), 404
        
        # Initialize detector
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        detector = FallDetectorMain(
            detector_type=model_config['detector_type'],
            model_path=model_config['model_path'],
            device=device,
            sequence_length=64,
        )
        
        # Process video
        result = detector.process_video(
            video_path=video_path,
            max_frames=None,  # Process all frames
            skip_frames=1,    # Process every frame
        )
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'model': model_id,
            'model_name': model_config['name'],
            'prediction': int(result['prediction']),
            'label': 'FALL DETECTED' if result['prediction'] == 1 else 'NO FALL',
            'confidence': float(result['probability']),
            'device': device,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add individual predictions for ensemble
        if 'individual_predictions' in result:
            response['individual_predictions'] = {
                name: {
                    'prediction': int(pred),
                    'confidence': float(conf),
                    'label': 'FALL' if pred == 1 else 'NO FALL',
                }
                for name, (pred, conf) in result['individual_predictions'].items()
            }
        
        # Save result to file
        result_filename = f"{filename.rsplit('.', 1)[0]}_{model_id}_result.json"
        result_path = RESULTS_FOLDER / result_filename
        with open(result_path, 'w') as f:
            json.dump(response, f, indent=2)
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in detect_fall: {error_trace}")
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'trace': error_trace,
        }), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup():
    """Clean up uploaded files and results."""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if filename:
            # Remove specific file
            video_path = UPLOAD_FOLDER / filename
            if video_path.exists():
                video_path.unlink()
            
            # Remove associated results
            result_pattern = f"{filename.rsplit('.', 1)[0]}_*_result.json"
            for result_file in RESULTS_FOLDER.glob(result_pattern):
                result_file.unlink()
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'cuda_available': torch.cuda.is_available(),
        'available_models': len(get_available_models()),
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413


if __name__ == '__main__':
    print("=" * 80)
    print("Fall Detection Web Application")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"\nAvailable models:")
    for model in get_available_models():
        print(f"  - {model['name']}: {model['description']}")
    print("=" * 80)
    print("\nStarting server on http://127.0.0.1:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
