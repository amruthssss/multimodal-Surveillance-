# 🚀 Quick Setup Guide for New Laptop (RTX GPU)

Follow these steps to get the surveillance system running on a new laptop with RTX GPU.

---

## ⏱️ Estimated Setup Time: 30-45 minutes

---

## 📋 Step-by-Step Installation

### 1️⃣ Install Python (5 minutes)

1. Download Python 3.10: https://www.python.org/downloads/
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
```bash
python --version
```
Should show: `Python 3.10.x`

---

### 2️⃣ Install Node.js (3 minutes)

1. Download Node.js LTS: https://nodejs.org/
2. Install with default settings
3. Verify:
```bash
node --version
npm --version
```

---

### 3️⃣ Install CUDA Toolkit (10 minutes)

1. Download CUDA 12.1: https://developer.nvidia.com/cuda-downloads
2. Select your OS and GPU
3. Install with default settings
4. Verify:
```bash
nvcc --version
nvidia-smi
```

---

### 4️⃣ Install Git (2 minutes)

1. Download: https://git-scm.com/downloads
2. Install with default settings

---

### 5️⃣ Clone Project (2 minutes)

```bash
# Open PowerShell or Command Prompt
git clone https://github.com/amruthssss/yolo.git
cd yolo
```

---

### 6️⃣ Setup Python Environment (8 minutes)

**Option A: Conda (Recommended)**
```bash
# Install Anaconda first: https://www.anaconda.com/download

# Create environment
conda create -n surveillance python=3.10 -y
conda activate surveillance

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt
```

**Option B: venv**
```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

---

### 7️⃣ Verify GPU Setup (1 minute)

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

✅ Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

### 8️⃣ Setup MongoDB Atlas (5 minutes)

1. Go to: https://www.mongodb.com/cloud/atlas
2. Click "Try Free"
3. Create account (use Google sign-in for speed)
4. Create FREE M0 cluster (select closest region)
5. Wait 3-5 minutes for cluster creation
6. Click "Database Access" → "Add New Database User"
   - Username: `surveillance`
   - Password: `YourPassword123` (save this!)
7. Click "Network Access" → "Add IP Address" → "Allow Access from Anywhere"
8. Click "Connect" → "Connect your application"
9. Copy connection string:
```
mongodb+srv://surveillance:<password>@cluster0.xxxxx.mongodb.net/surveillance
```

---

### 9️⃣ Configure Backend (3 minutes)

```bash
cd surveillance-backend
```

Create/Edit `.env` file:
```env
PORT=5001
MONGODB_URI=mongodb+srv://surveillance:YourPassword123@cluster0.xxxxx.mongodb.net/surveillance?retryWrites=true&w=majority
JWT_SECRET=super_secret_key_change_this
JWT_EXPIRE=7d
```

Install dependencies:
```bash
npm install
```

---

### 🔟 Configure Frontend (3 minutes)

```bash
cd ../surveillance-frontend
npm install
```

---

## 🎉 Launch Application (1 minute)

```bash
# Return to project root
cd ..

# Run startup script
.\START.bat
```

This will:
- ✅ Start Backend (port 5001)
- ✅ Start Frontend (port 3000)
- ✅ Open browser automatically

---

## 🎯 First Login

1. Browser opens to: `http://localhost:3000`
2. Click **"GET STARTED"**
3. Click **"REGISTER"** tab
4. Fill in details:
   - Username: `admin`
   - Email: `admin@test.com`
   - Mobile: `9876543210`
   - Password: `admin123`
5. Click **"CREATE ACCOUNT"**
6. **Check backend terminal** for OTP (6 digits)
7. Enter OTP
8. Login with username/password
9. You're in! 🎉

---

## 🎥 Test Camera

1. Go to **Live Feeds** tab
2. Click **⚙️** (settings icon)
3. Select **"Built-in Camera"**
4. Click **"SAVE"**
5. Allow browser camera permission
6. You should see yourself! 📹

---

## 🚨 Common Issues & Fixes

### ❌ CUDA Not Available

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### ❌ MongoDB Connection Failed

- Check internet connection
- Verify password in `.env` file
- Ensure IP whitelist includes 0.0.0.0/0

### ❌ Port Already in Use

```bash
# Kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID_NUMBER> /F

# Kill process on port 5001
netstat -ano | findstr :5001
taskkill /PID <PID_NUMBER> /F
```

### ❌ npm install fails

```bash
# Clear cache
npm cache clean --force
npm install
```

---

## 📊 Performance Check

Run this to verify everything works:

```bash
# Activate environment
conda activate surveillance

# Test detection
python enhanced_final_ultra_system.py
```

Should show:
- ✅ CUDA enabled
- ✅ GPU detected
- ✅ Models loaded
- ✅ FPS: 25-30

---

## 🎮 GPU Performance by Model

| GPU | FPS | Latency | Memory |
|-----|-----|---------|--------|
| RTX 4060 | 28-32 | 35ms | 4-6 GB |
| RTX 3060 | 25-30 | 40ms | 5-7 GB |
| RTX 2060 | 20-25 | 50ms | 6-8 GB |

---

## ✅ Checklist

Before closing:

- [ ] Python 3.10 installed
- [ ] Node.js installed
- [ ] CUDA Toolkit installed
- [ ] Git installed
- [ ] Project cloned
- [ ] Python environment created
- [ ] PyTorch CUDA works
- [ ] Dependencies installed
- [ ] MongoDB Atlas configured
- [ ] Backend `.env` configured
- [ ] npm packages installed
- [ ] Servers start successfully
- [ ] Can register and login
- [ ] Camera works

---

## 🆘 Need Help?

1. Check main README.md
2. Open GitHub issue
3. Review troubleshooting section

---

**⏱️ Total Time: ~30-45 minutes**

**🎉 Happy Monitoring!**
