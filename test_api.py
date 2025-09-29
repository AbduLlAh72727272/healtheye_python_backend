#!/usr/bin/env python3
"""
Quick test script for the deployed API
"""

import requests
import base64
from PIL import Image
import io
import numpy as np

# Create a minimal test image
img = Image.new('RGB', (100, 100), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

# Convert to base64
b64_image = base64.b64encode(img_bytes.read()).decode('utf-8')

# Test the deployed API
url = "https://healtheye-python-backend-3.onrender.com/predict"
data = {"image": b64_image}

try:
    response = requests.post(url, json=data, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            top_pred = result['result']['top_5_predictions'][0]
            print(f"\nTop Prediction:")
            print(f"Label: {top_pred['label']}")
            print(f"Confidence: {top_pred['confidence']:.4f}")
            print(f"Percentage: {top_pred['percentage']:.2f}%")
            
            print(f"\nAll Top 5:")
            for i, pred in enumerate(result['result']['top_5_predictions'][:5]):
                print(f"{i+1}. {pred['label']}: {pred['percentage']:.2f}%")
        else:
            print(f"Prediction failed: {result.get('error', 'Unknown error')}")
    else:
        print(f"HTTP Error: {response.text}")
        
except Exception as e:
    print(f"Request failed: {e}")