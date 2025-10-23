# Complete Dataset Transfer & Preparation Guide

This guide helps you prepare a complete 4-class YOLO dataset (fighting, vehicle_accident, fire, explosion) for training.

## Current Status

Your `yolo_dataset_fixed.zip` contains:
- ✅ fighting
- ✅ vehicle_accident (accidents)
- ❌ fire (missing)
- ❌ explosion (missing)

## Goal

Create a complete dataset with all 4 classes in `yolo_ready/` folder.

---

## Step-by-Step Workflow

### Step 1: Check what you have

```powershell
# Extract your current dataset
Expand-Archive -Path .\yolo_dataset_fixed.zip -DestinationPath .\datasets -Force

# Scan to see class distribution
python prepare_complete_dataset.py --scan --dataset-root datasets\yolo_ready
```

Expected output:
```
📈 Class distribution:
  ❌ fire: 0 instances
  ✅ vehicle_accident: 1234 instances
  ✅ fighting: 5678 instances
  ❌ explosion: 0 instances
```

---

### Step 2: Get fire & explosion videos

You need video footage of fire and explosion events. Here are some options:

#### Option A: Download from public datasets

**Fire videos:**
- Kaggle: [Fire Detection Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)
- GitHub: Search "fire detection dataset video"
- YouTube: Search "fire cctv footage" (use youtube-dl to download, ensure license allows)

**Explosion videos:**
- Kaggle: Search "explosion detection"
- Public CCTV footage repositories
- YouTube: "explosion cctv" or "industrial accident footage"

#### Option B: Use your own footage

If you have access to CCTV footage with fire/explosion events, use those.

---

### Step 3: Organize your videos

Create folders and place videos:

```powershell
# Create folders
New-Item -ItemType Directory -Path .\fire_videos -Force
New-Item -ItemType Directory -Path .\explosion_videos -Force

# Copy/move your fire videos to fire_videos/
# Copy/move your explosion videos to explosion_videos/
```

Expected structure:
```
D:\muli_modal\
├── fire_videos\
│   ├── fire_scene_01.mp4
│   ├── fire_scene_02.mp4
│   └── ...
├── explosion_videos\
│   ├── explosion_01.mp4
│   ├── explosion_02.mp4
│   └── ...
└── datasets\
    └── yolo_ready\
        ├── images\
        └── labels\
```

---

### Step 4: Add fire class to dataset

```powershell
python prepare_complete_dataset.py --add-class fire --videos fire_videos --dataset-root datasets\yolo_ready --max-frames 150
```

This will:
- Extract frames from fire videos using motion detection
- Generate YOLO bounding box labels (class_id = 0 for fire)
- Add images to `datasets/yolo_ready/images/train` and `/val`
- Add labels to `datasets/yolo_ready/labels/train` and `/val`

---

### Step 5: Add explosion class to dataset

```powershell
python prepare_complete_dataset.py --add-class explosion --videos explosion_videos --dataset-root datasets\yolo_ready --max-frames 150
```

This will:
- Extract frames from explosion videos
- Generate YOLO labels (class_id = 3 for explosion)
- Merge into the same train/val splits

---

### Step 6: Verify all classes present

```powershell
python prepare_complete_dataset.py --scan --dataset-root datasets\yolo_ready
```

Expected output:
```
📈 Class distribution:
  ✅ fire: 450 instances
  ✅ vehicle_accident: 1234 instances
  ✅ fighting: 5678 instances
  ✅ explosion: 380 instances

✅ All 4 classes present!
```

---

### Step 7: Finalize dataset (create data.yaml)

```powershell
python prepare_complete_dataset.py --finalize --dataset-root datasets\yolo_ready
```

This creates `datasets/yolo_ready/data.yaml` with:
```yaml
train: D:/muli_modal/datasets/yolo_ready/images/train
val: D:/muli_modal/datasets/yolo_ready/images/val
nc: 4
names: ['fire', 'vehicle_accident', 'fighting', 'explosion']
```

---

### Step 8: Create new zip and transfer to laptop

```powershell
# Create updated zip with all 4 classes
Compress-Archive -Path .\datasets\yolo_ready\* -DestinationPath .\yolo_dataset_complete.zip -Force

# Check size
Get-Item yolo_dataset_complete.zip | Select-Object Name, Length
```

Transfer options:
- **USB drive**: Copy `yolo_dataset_complete.zip` to USB, then to laptop
- **Network share**: `Copy-Item yolo_dataset_complete.zip \\laptop-ip\share\`
- **Google Drive**: Upload and download on laptop

---

## Quick Commands for Laptop (RTX 4060)

After transferring the zip to the laptop:

```powershell
# 1. Clone training repo
git clone https://github.com/amruthssss/yolo.git
cd yolo

# 2. Extract dataset
Expand-Archive -Path ..\yolo_dataset_complete.zip -DestinationPath .\yolo_ready -Force

# 3. Verify dataset
python verify_dataset.py

# 4. Create conda env
conda env create -f environment.yml
conda activate yolov11-train
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 5. Start training
.\start_training.ps1 -DataPath .\yolo_ready -Model yolov11m.pt -Epochs 100 -Batch 8 -Accumulate 2 -AutoCreate
```

---

## Troubleshooting

### "No frames extracted (no motion detected)"
- Videos might have low motion or static frames
- Adjust `MOTION_THRESHOLD` (lower = more sensitive) in `prepare_complete_dataset.py`
- Or manually extract key frames using a video editor

### "Class distribution still unbalanced"
- Add more videos for underrepresented classes
- Use `--max-frames 200` to extract more frames per video
- Consider data augmentation during training (already enabled in train script)

### "Videos too large to transfer"
- Compress videos before processing: `ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4`
- Or process videos on current PC, then transfer only the extracted `yolo_ready/` folder

---

## Optional: Manual Frame Extraction (if script fails)

If the automated script doesn't work for your videos:

```powershell
# Extract frames manually using ffmpeg
ffmpeg -i fire_video.mp4 -vf "select='gt(scene,0.3)',fps=2" -vsync vfr fire_frames\frame_%04d.jpg

# Then manually create YOLO labels or use labeling tools like:
# - labelImg (https://github.com/HumanSignal/labelImg)
# - CVAT (https://www.cvat.ai/)
# - Roboflow (https://roboflow.com/)
```

---

## Summary Checklist

- [ ] Extract current dataset and scan
- [ ] Download/collect fire videos → place in `fire_videos/`
- [ ] Download/collect explosion videos → place in `explosion_videos/`
- [ ] Run `--add-class fire --videos fire_videos`
- [ ] Run `--add-class explosion --videos explosion_videos`
- [ ] Run `--scan` to verify all 4 classes present
- [ ] Run `--finalize` to create `data.yaml`
- [ ] Create new zip: `yolo_dataset_complete.zip`
- [ ] Transfer zip to laptop (USB/network/Drive)
- [ ] Extract on laptop and run training

---

**Need help?** Check the error messages from `prepare_complete_dataset.py` or ask for assistance!
