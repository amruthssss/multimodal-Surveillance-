# ⚡ QUICK START - For Your Other Laptop

**Already have Python 3.10, Node.js, and CUDA installed? Start here!**

---

## 🚀 3-Step Setup (5 Minutes)

### Step 1: Python Environment
```powershell
# In project folder (D:\muli_modal)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2: Node Dependencies
```powershell
cd surveillance-backend
npm install
cd ..\surveillance-frontend
npm install
cd ..
```

### Step 3: Run Everything
```powershell
.\START.bat
```

---

## 🌐 Run Commands

### Automated (Easiest):
```powershell
.\START.bat
```
Opens browser at: `http://localhost:3000`

### Manual (3 Terminals):

**Terminal 1 - Backend:**
```powershell
cd surveillance-backend
npm start
```

**Terminal 2 - Frontend:**
```powershell
cd surveillance-frontend
npm start
```

**Terminal 3 - AI Engine:**
```powershell
.\venv\Scripts\Activate.ps1
python app.py
```

---

## 🎯 Access Website

**URL:** `http://localhost:3000`

**Flow:**
1. Landing Page → Click "GET STARTED"
2. Sign Up → Enter details
3. OTP → Check backend terminal for 6-digit code
4. Login → Use credentials
5. Dashboard → Configure camera & start monitoring

---

## 📹 Camera Setup

**Built-in Webcam:**
- Dashboard → Live Feeds → ⚙️ Configure Camera
- Select "Built-in Camera" → SAVE

**RTSP Stream:**
- Enter: `rtsp://username:password@ip:port/stream`

**Video File:**
- Enter full path: `D:\Videos\video.mp4`

---

## ✅ Verify GPU
```powershell
.\venv\Scripts\Activate.ps1
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

**Expected:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

---

## 🔧 Troubleshooting

**Port in use:**
```powershell
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**CUDA not working:**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**MongoDB error:**
- Check internet connection
- `.env` is pre-configured, no changes needed

---

## 📊 Performance (RTX 4060)

- **FPS:** 28-32
- **GPU Memory:** 4-6 GB
- **Latency:** 35ms
- **Accuracy:** 95%+

---

**That's it! The system is pre-configured for your setup.**

*No .env changes needed | No MongoDB setup needed | Just copy & run*
