#!/bin/bash
# Render Deployment Script - Multiple Fallbacks
# This script tries different requirements files until one works

echo "🚀 Starting HealthEye Python Backend Deployment"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# Upgrade pip first
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Try requirements files in order of preference
echo "📋 Trying requirements files..."

if pip install -r requirements.txt; then
    echo "✅ Successfully installed from requirements.txt (TensorFlow)"
    echo "DEPLOYMENT_TYPE=tensorflow" > deployment_info.txt
elif pip install -r requirements-lite.txt; then
    echo "✅ Successfully installed from requirements-lite.txt (TensorFlow Lite)"
    echo "DEPLOYMENT_TYPE=tflite" > deployment_info.txt
elif pip install -r requirements-minimal.txt; then
    echo "✅ Successfully installed from requirements-minimal.txt (Minimal)"
    echo "DEPLOYMENT_TYPE=minimal" > deployment_info.txt
else
    echo "❌ All requirements files failed!"
    exit 1
fi

echo "🎉 Build completed successfully!"
echo "📊 Installed packages:"
pip list