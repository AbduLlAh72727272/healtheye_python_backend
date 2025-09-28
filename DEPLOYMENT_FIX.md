# 🚨 Python Backend Deployment - Troubleshooting Guide

## ✅ **FIXED: TensorFlow Compatibility Issue**

**Problem**: TensorFlow 2.19.1 not compatible with Python 3.13
**Solution**: Updated to TensorFlow 2.15.0 + Python 3.11.9

---

## 🐍 **Updated Deployment Instructions**

### **Files Updated:**
1. ✅ `requirements.txt` - Compatible TensorFlow version
2. ✅ `runtime.txt` - Specifies Python 3.11.9
3. ✅ `render.yaml` - Enhanced build configuration
4. ✅ `requirements-lite.txt` - Backup lightweight version

### **New Build Command:**
```bash
pip install --upgrade pip && pip install -r requirements.txt
```

### **Enhanced Start Command:**
```bash
gunicorn --bind 0.0.0.0:$PORT --timeout 120 --workers 1 prediction_server:app
```

---

## 🔄 **How to Redeploy on Render**

### **Option 1: Auto Redeploy**
1. Commit and push these changes to GitHub
2. Render will automatically redeploy (if auto-deploy is enabled)

### **Option 2: Manual Redeploy**
1. Go to your Python service in Render dashboard
2. Click **"Manual Deploy"** → **"Deploy latest commit"**

---

## 📊 **If Build Still Fails - Use Lightweight Version**

If TensorFlow is still too heavy for free tier:

### **Step 1:** Rename files
```bash
mv requirements.txt requirements-full.txt
mv requirements-lite.txt requirements.txt
```

### **Step 2:** Update Render build command to:
```bash
pip install --upgrade pip && pip install -r requirements.txt
```

### **Step 3:** Redeploy

---

## 🔍 **Monitoring Your Deployment**

### **Check Build Logs:**
1. Go to Render dashboard
2. Click on your Python service
3. Check "Logs" tab for build progress

### **Expected Build Time:**
- ✅ With TensorFlow: 5-8 minutes
- ✅ With TensorFlow Lite: 2-3 minutes

### **Memory Usage:**
- ✅ TensorFlow: ~400-500MB
- ✅ TensorFlow Lite: ~200-300MB

---

## 🚀 **Deployment Success Indicators**

### **Build Success:**
```
==> Build succeeded 🎉
==> Your service is live at https://healtheye-python-backend.onrender.com
```

### **Health Check:**
```bash
curl https://healtheye-python-backend.onrender.com/health
# Should return: {"status": "healthy", ...}
```

### **Model Loading:**
Check logs for:
```
Model loaded successfully using: tensorflow/tflite-runtime
TensorFlow backend: tensorflow/tflite-runtime
```

---

## ⚠️ **Common Issues & Solutions**

### **Issue 1: Memory Exceeded**
**Error**: `Build failed: out of memory`
**Solution**: Switch to `requirements-lite.txt`

### **Issue 2: Build Timeout**
**Error**: `Build timed out`
**Solution**: Build command already includes pip upgrade

### **Issue 3: Model Not Found**
**Error**: `Model file not found`
**Solution**: Ensure your `.tflite` model file is in the repository

### **Issue 4: Cold Start Issues**
**Error**: First request times out
**Solution**: Increased gunicorn timeout to 120 seconds

---

## 📈 **Performance Optimizations Applied**

1. ✅ **Python Version**: 3.11.9 (optimal for TensorFlow)
2. ✅ **TensorFlow**: 2.15.0 (stable, compatible)
3. ✅ **Gunicorn**: Single worker (memory efficient)
4. ✅ **Timeout**: 120 seconds (handles model loading)
5. ✅ **Alternative**: TensorFlow Lite option for lighter deployment

---

## 🎉 **Next Steps After Successful Deployment**

1. ✅ Test the health endpoint
2. ✅ Test prediction endpoint with sample image
3. ✅ Update your Flutter app with the new URL
4. ✅ Monitor resource usage in Render dashboard

**Your Python API will be live at:**
```
https://healtheye-python-backend.onrender.com
```