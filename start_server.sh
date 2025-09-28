#!/bin/bash
# Startup script for HealthEye Python Backend
# Handles dependency installation and server startup

echo "🚀 Starting HealthEye Python Backend..."

# Check Python version
echo "📋 Python version: $(python --version)"
echo "📋 pip version: $(pip --version)"

# Upgrade pip and essential tools
echo "🔧 Upgrading pip and build tools..."
python -m pip install --upgrade pip setuptools wheel

# Try to install dependencies with multiple fallbacks
echo "📦 Installing dependencies..."

if pip install --no-cache-dir -r requirements.txt; then
    echo "✅ Main requirements installed successfully"
elif pip install --no-cache-dir -r requirements-lite.txt; then
    echo "✅ Lite requirements installed successfully"  
elif pip install --no-cache-dir -r requirements-minimal.txt; then
    echo "✅ Minimal requirements installed successfully"
else
    echo "⚠️ Installing ultra-minimal dependencies..."
    pip install --no-cache-dir Flask flask-cors gunicorn Pillow requests python-dotenv
fi

# Install any missing core packages individually
echo "🔍 Checking core dependencies..."
python -c "import flask" || pip install --no-cache-dir Flask
python -c "import flask_cors" || pip install --no-cache-dir flask-cors
python -c "import gunicorn" || pip install --no-cache-dir gunicorn
python -c "from PIL import Image" || pip install --no-cache-dir Pillow

# Check if TensorFlow is available
echo "🧠 Checking ML dependencies..."
python -c "import tensorflow" && echo "✅ TensorFlow available" || echo "⚠️ TensorFlow not available - running in minimal mode"
python -c "import numpy" && echo "✅ NumPy available" || echo "⚠️ NumPy not available"

# Start the server
echo "🌟 Starting server..."
exec gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --max-requests 1000 --preload prediction_server:app