#!/usr/bin/env python3
"""
TensorFlow Model Prediction Server for HealthEye - FAST STARTUP VERSION
Optimized for Render deployment with lazy model loading
"""

import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import json
import threading
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Try to import TensorFlow with fallbacks
TF_AVAILABLE = False
TFLITE_AVAILABLE = False

try:
    import tensorflow as tf
    # Suppress deprecated interpreter warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*tf.lite.Interpreter is deprecated.*')
    TF_AVAILABLE = True
    print("✅ TensorFlow imported successfully")
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    tf = None

app = Flask(__name__)
CORS(app)

# Model configuration
IMG_SIZE = (224, 224)
NUM_CLASSES = 23

# Class labels
class_labels = [
    'barretts', 'barretts-short-segment', 'bbps-0-1', 'bbps-2-3', 'cecum',
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis-a', 'esophagitis-b-d',
    'hemorrhoids', 'ileum', 'impacted-stool', 'polyps', 'pylorus', 'retroflex-rectum',
    'retroflex-stomach', 'ulcerative-colitis-grade-0-1', 'ulcerative-colitis-grade-1',
    'ulcerative-colitis-grade-1-2', 'ulcerative-colitis-grade-2', 'ulcerative-colitis-grade-2-3',
    'ulcerative-colitis-grade-3', 'z-line'
]

# Global model variables
model = None
model_loaded = False
model_error = None
model_loading = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use TensorFlow Lite
try:
    TFLiteInterpreter = tf.lite.Interpreter
    TFLITE_BACKEND = 'tensorflow'
except Exception:
    TFLiteInterpreter = None
    TFLITE_BACKEND = 'none'

def preprocess_image(image_bytes):
    """Fast image preprocessing"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {e}")

def load_model_async():
    """Load model asynchronously to avoid startup timeout"""
    global model, model_loaded, model_error, model_loading
    
    if model_loading:
        return
    
    model_loading = True
    
    try:
        print("🔄 Starting async model loading...")
        
        # Model search paths
        search_paths = [
            os.path.join(BASE_DIR, 'assets', 'models', 'model.tflite'),
            os.path.join(BASE_DIR, 'assets', 'models', 'model_32.tflite'),
            os.path.join(BASE_DIR, 'model.tflite'),
            os.path.join(BASE_DIR, '..', 'assets', 'models', 'model.tflite'),
        ]
        
        model_path = None
        for path in search_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(f"Model file not found. Searched: {search_paths}")

        print(f"📂 Loading model from: {model_path}")
        
        # Load with timeout protection
        interpreter = TFLiteInterpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        model = {
            'interpreter': interpreter,
            'input_details': input_details,
            'output_details': output_details,
            'model_path': model_path
        }
        
        model_loaded = True
        model_error = None
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        model_error = str(e)
        model_loaded = False
        print(f"❌ Model loading failed: {e}")
    finally:
        model_loading = False

def predict_image(img_array):
    """Make prediction using the loaded model"""
    if not model_loaded or model is None:
        raise Exception("Model not ready")
    
    try:
        interpreter = model['interpreter']
        input_details = model['input_details']
        output_details = model['output_details']
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        return predictions[0]  # Remove batch dimension
        
    except Exception as e:
        raise Exception(f"Prediction failed: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Fast health check - doesn't wait for model"""
    global model_loading
    
    # Start model loading if not started
    if not model_loaded and not model_loading and not model_error:
        thread = threading.Thread(target=load_model_async)
        thread.daemon = True
        thread.start()
    
    status = "loading" if model_loading else ("ready" if model_loaded else "error")
    
    return jsonify({
        'status': 'healthy',
        'service': 'HealthEye Prediction API - FAST STARTUP',
        'model_status': status,
        'model_loaded': model_loaded,
        'model_loading': model_loading,
        'model_error': model_error if not model_loading else None,
        'can_predict': model_loaded,
        'production_mode': True,
        'mock_mode': False,
        'tensorflow_available': TF_AVAILABLE,
        'deployment_type': 'production-fast-startup',
        'model_info': {
            'num_classes': NUM_CLASSES,
            'input_size': IMG_SIZE,
            'class_labels_count': len(class_labels),
            'backend': TFLITE_BACKEND if model_loaded else 'loading'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint with model loading check"""
    try:
        # Start model loading if needed
        if not model_loaded and not model_loading and not model_error:
            thread = threading.Thread(target=load_model_async)
            thread.daemon = True
            thread.start()
        
        # Wait for model to load (with timeout)
        wait_time = 0
        max_wait = 30  # 30 seconds max wait
        
        while model_loading and wait_time < max_wait:
            time.sleep(0.5)
            wait_time += 0.5
        
        if not model_loaded:
            if model_loading:
                return jsonify({
                    'success': False,
                    'error': 'Model is still loading, please try again in a few seconds'
                }), 503
            else:
                return jsonify({
                    'success': False,
                    'error': f'Model failed to load: {model_error}'
                }), 500
        
        # Get and validate image
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400
        
        try:
            image_bytes = base64.b64decode(data['image'])
            img_array = preprocess_image(image_bytes)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Image processing failed: {e}'
            }), 400
        
        # Make prediction
        try:
            predictions = predict_image(img_array)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Prediction failed: {e}'
            }), 500
        
        # Format results
        top_indices = np.argsort(predictions)[::-1]
        top_5_predictions = []
        
        for i in range(min(5, len(top_indices))):
            idx = top_indices[i]
            confidence = float(predictions[idx])
            label = class_labels[idx] if idx < len(class_labels) else f"Class_{idx}"
            
            top_5_predictions.append({
                'label': label,
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        top_prediction = top_5_predictions[0] if top_5_predictions else None
        
        return jsonify({
            'success': True,
            'result': {
                'prediction': top_prediction['label'] if top_prediction else 'unknown',
                'confidence': top_prediction['confidence'] if top_prediction else 0.0,
                'percentage': top_prediction['percentage'] if top_prediction else 0.0,
                'top_5_predictions': top_5_predictions,
                'model_info': {
                    'preprocessing': 'resize_224x224_normalize_0_1',
                    'model_type': 'EfficientNetB0-Compatible (PRODUCTION)',
                    'num_classes': NUM_CLASSES,
                    'production_mode': True,
                    'mock_mode': False
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {e}'
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded yet',
            'model_status': 'loading' if model_loading else 'error'
        }), 503
    
    return jsonify({
        'success': True,
        'model_info': {
            'path': model.get('model_path', 'unknown'),
            'input_shape': model['input_details'][0]['shape'].tolist(),
            'output_shape': model['output_details'][0]['shape'].tolist(),
            'num_classes': NUM_CLASSES,
            'class_labels': class_labels,
            'backend': TFLITE_BACKEND,
            'production_mode': True
        }
    })

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - starts fast"""
    return jsonify({
        'message': '🏥 HealthEye Image Classification API - FAST STARTUP',
        'version': '2.1.0',
        'status': 'running',
        'model_status': 'loading' if model_loading else ('ready' if model_loaded else 'initializing'),
        'endpoints': {
            'health': 'GET /health',
            'predict': 'POST /predict',
            'model_info': 'GET /model/info'
        }
    })

# DO NOT load model on import - let it load async on first request
print("🚀 HealthEye Fast Startup Server - Ready!")
print("⚡ Model will load on first request to avoid deployment timeout")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)), debug=False)