#!/usr/bin/env python3
"""
HealthEye Python Backend - App Entry Point
Simple wrapper for prediction_server.py for Render deployment
"""

import os

try:
    from prediction_server import app
    print("✅ Successfully imported Flask app from prediction_server")
except ImportError as e:
    print(f"❌ Failed to import Flask app: {e}")
    # Create a minimal Flask app as fallback
    from flask import Flask, jsonify
    app = Flask(__name__)
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'mode': 'fallback'})
    
    @app.route('/')
    def root():
        return jsonify({'message': 'HealthEye Backend - Fallback Mode'})

if __name__ == '__main__':
    # This allows the app to run directly for testing
    # In production, Gunicorn will import the 'app' object
    print("🚀 Starting HealthEye Backend...")
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)), debug=False)