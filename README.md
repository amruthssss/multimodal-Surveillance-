#  Next-Gen AI Surveillance System

**Advanced Multi-Modal AI-Powered Real-Time Security Monitoring System**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2.0-61dafb)](https://reactjs.org/)
[![Node.js](https://img.shields.io/badge/Node.js-22.x-green)](https://nodejs.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)](https://developer.nvidia.com/cuda-toolkit)

---

##  Quick Links
- [ Overview](#-overview)
- [ Features](#-features)  
- [ System Architecture](#-system-architecture)
- [ Technology Stack](#-technology-stack)
- [ Prerequisites](#-prerequisites)
- [ Installation](#-installation)
- [ Quick Start](#-quick-start)
- [ Troubleshooting](#-troubleshooting)

---

##  Overview

A cutting-edge AI surveillance system combining **YOLOv11m** object detection with a **Hybrid AI Agent** for real-time security monitoring with advanced threat detection capabilities.

### What This System Detects:
-  **Fire & Explosion** (flame color analysis)
-  **Road Accidents** (vehicle collision tracking)  
-  **Violence & Fighting** (action recognition)
-  **Suspicious Activities** (behavior patterns)
-  **Emotion Analysis** (facial expressions)
-  **Audio Anomalies** (screams, glass breaking)

### Key Features:
-  **Temporal Accuracy**: Detects events DURING occurrence
-  **False Positive Prevention**: 95%+ accuracy
-  **Multi-Modal Fusion**: Visual + Audio + Motion
-  **Pattern Learning**: 221,660+ detection patterns
-  **Real-Time**: 30 FPS with CUDA acceleration

---

##  Features

###  Multi-Source Camera Support
- Built-in Webcam | RTSP Streams | IP Cameras | YouTube Live | Video Files

###  AI Detection Models  
- **YOLOv11m**: 20M parameters (accident detection)
- **YOLOv8n**: 3M parameters (object detection)
- **Hybrid Agent**: 80% Pattern + 20% YOLO  
- **Emotion CNN**: 7-class facial expression

###  Smart Alert System
- Email/SMS notifications
- Real-time dashboard updates (Socket.IO)
- Risk level classification (Low/Medium/High)

###  Modern Web Interface
- **Landing Page**: Futuristic dark theme
- **Authentication**: JWT + OTP verification
- **Dashboard**: Live Feeds, Alerts, Analytics, Configuration
- **Responsive**: Mobile and desktop optimized

---

##  System Architecture

"'

        React Frontend (Port 3000)       
   Landing  Auth  Dashboard            

             WebSocket + REST API

      Node.js Backend (Port 5001)        
   JWT Auth | OTP | MongoDB Atlas        


      Flask Backend (Port 5000)          
   Video Stream | AI Detection           


          AI Engine                      
   YOLOv11m + Hybrid Agent + Emotion     
   221,660 Learned Patterns              

"'

---

##  Technology Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18.2.0, Socket.IO, Axios |
| **Backend** | Node.js + Express, Flask 3.0 |
| **Database** | MongoDB Atlas, SQLite |
| **AI Models** | YOLOv11m, YOLOv8n, TensorFlow |
| **Computer Vision** | OpenCV 4.8.1, Ultralytics |
| **GPU** | CUDA 12.x, PyTorch 2.1 |

---

##  Prerequisites

### Hardware
- **GPU**: NVIDIA RTX 2060+ (RTX 4060 recommended)
- **RAM**: 16 GB minimum (32 GB recommended)
- **Storage**: 5 GB free space

### Software
1. **Python 3.8-3.11** (3.10 recommended) - https://python.org
2. **Node.js 18+** (22.x recommended) - https://nodejs.org  
3. **CUDA Toolkit 12.x** - https://developer.nvidia.com/cuda-downloads
4. **cuDNN 8.x** - https://developer.nvidia.com/cudnn
5. **Git** - https://git-scm.com
6. **MongoDB Atlas** (free) - https://mongodb.com/cloud/atlas

---

##  Installation

### Step 1: Clone Repository
"'bash
git clone https://github.com/amruthssss/yolo.git
cd yolo
"'

### Step 2: Python Environment

**Conda (Recommended):**
"'bash
conda create -n surveillance python=3.10 -y
conda activate surveillance
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
"'

**venv:**
"'bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
"'

### Step 3: Verify GPU
"'bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
"'

### Step 4: MongoDB Atlas Setup
1. Create free account: https://mongodb.com/cloud/atlas
2. Create M0 cluster (FREE)
3. Add Database User (username/password)
4. Network Access  Allow 0.0.0.0/0
5. Connect  Copy connection string

### Step 5: Backend Configuration
"'bash
cd surveillance-backend
"'

Create .env file:
"'env
PORT=5001
MONGODB_URI=mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/surveillance
JWT_SECRET=your_secret_key_change_this
JWT_EXPIRE=7d
"'

"'bash
npm install
"'

### Step 6: Frontend Setup
"'bash
cd ../surveillance-frontend
npm install
"'

---

##  Quick Start

### Automated Startup (Recommended)

**Windows:**
"'bash
.\START.bat
"'

**PowerShell:**
"'powershell
.\START.ps1
"'

This automatically:
-  Checks Node.js installation
-  Installs dependencies
-  Starts backend (port 5001)
-  Starts frontend (port 3000)
-  Opens browser

### Manual Startup

**Terminal 1 - Backend:**
"'bash
cd surveillance-backend
npm start
"'

**Terminal 2 - Frontend:**
"'bash
cd surveillance-frontend
npm start
"'

**Terminal 3 - Python AI:**
"'bash
conda activate surveillance
python app.py
"'

---

##  Usage Guide

### First Time Setup

1. Open: http://localhost:3000
2. Click **"GET STARTED"**
3. **Register**:
   - Username (3+ chars)
   - Email (valid format)
   - Mobile (10 digits)
   - Password (6+ chars)
4. **OTP**: Check backend terminal for 6-digit code
5. **Login**: Use credentials

### Configure Camera

1. Dashboard  **Live Feeds** tab
2. Click ** Settings**
3. Select source:
   - Built-in Camera
   - RTSP: tsp://username:password@ip:port/stream
   - IP Camera: http://192.168.1.100/video
   - YouTube Live
   - Video File: D:\path\to\video.mp4
4. Click **SAVE**

---

##  Project Structure

"'
yolo/
 surveillance-backend/     # Node.js Backend
    models/              # MongoDB schemas
    routes/              # API endpoints
    middleware/          # JWT auth
    server.js            # Express app
 surveillance-frontend/   # React Frontend  
    src/
       pages/          # Landing, Auth, Dashboard
       context/        # Global state
    public/
 models/                  # AI Model Weights
    yolov8n.pt
    emotion_model.h5
    runs/detect/train/weights/best.pt
 utils/                   # Python Utilities
    yolo_wrapper.py
    emotion_wrapper.py
    fusion_inference.py
 enhanced_final_ultra_system.py  # Main AI System
 app.py                   # Flask Server
 requirements.txt         # Python deps
 START.bat               # Startup script
"'

---

##  Configuration

### Detection Thresholds

Edit enhanced_final_ultra_system.py:

"'python
MOTION_THRESHOLD = 180       # Motion sensitivity
FIRE_THRESHOLD = 0.05        # 5% flame colors
EXPLOSION_COLOR_THRESHOLD = 0.08
EXPLOSION_BRIGHTNESS_THRESHOLD = 0.15
ACCIDENT_MIN_VEHICLES = 0.30  # 30% vehicles
"'

---

##  Troubleshooting

### CUDA Not Available
"'bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
"'

### MongoDB Connection Failed
- Check internet connection
- Verify password in .env
- Ensure IP whitelist: 0.0.0.0/0

### Port Already in Use
"'bash
netstat -ano | findstr :3000
taskkill /PID <PID> /F
"'

### Camera Not Working
- Built-in: Allow browser permissions
- RTSP: Test with VLC player
- Video File: Use absolute path

---

##  Performance (RTX 4060)

| Metric | Value |
|--------|-------|
| FPS | 28-32 |
| Latency | 35ms |
| GPU Memory | 4-6 GB |
| Accuracy | 95%+ |

---

##  Documentation

- **SETUP_NEW_LAPTOP.md**: 30-minute setup guide
- **PROJECT_SUMMARY.md**: Detailed project explanation
- **README.md**: This file

---

##  Author

**Amruth**  
GitHub: [@amruthssss](https://github.com/amruthssss)

---

##  Acknowledgments

- Ultralytics (YOLO)
- OpenCV Community
- React & Node.js Communities
- MongoDB Atlas

---

** Star this project if you find it helpful!**

Last Updated: October 25, 2025
