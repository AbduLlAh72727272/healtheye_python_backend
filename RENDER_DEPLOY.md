# 🚀 HealthEye Python Backend - Render Deployment Guide

## 📋 Quick Deploy Steps

### 1. Push Code to GitHub
Make sure your code is pushed to your GitHub repository.

### 2. Connect to Render
1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository `healtheye`

### 3. Configure Service
- **Name**: `healtheye-python-backend`
- **Region**: Oregon (US-West)
- **Branch**: `main`
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT prediction_server:app`
- **Root Directory**: `backend_python`

### 4. Set Environment Variables
In Render dashboard, add:
```
FLASK_ENV = production
```

### 5. Deploy
Click **"Create Web Service"** and wait for deployment.

## 🔗 Your API will be available at:
```
https://healtheye-python-backend.onrender.com
```

## 📝 API Endpoints:
- `GET /health` - Health check
- `GET /` - API info
- `POST /predict` - Image classification

## 🛠️ Testing Your Deployment:
```bash
curl https://healtheye-python-backend.onrender.com/health
```

## ⚠️ Important Notes:
- TensorFlow model loading takes time on first startup
- Free tier has cold starts (may be slow on first request)
- Service sleeps after 15 minutes of inactivity
- 750 hours/month free tier limit
- Large ML models may hit memory limits on free tier