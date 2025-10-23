# Kaggle Dataset Download & Organization Guide

This guide shows how to download fire & explosion datasets from Kaggle on your RTX 4060 laptop and merge them with your existing fighting/accident datasets.

---

## Prerequisites (One-Time Setup)

### 1. Install Kaggle API

```powershell
pip install kaggle opencv-python
```

### 2. Get Kaggle API Credentials

1. Go to https://www.kaggle.com/account (create account if needed)
2. Scroll to **API** section
3. Click **"Create New API Token"**
4. Download `kaggle.json` (contains your credentials)
5. Place it in the correct location:

**Windows:**
```powershell
# Create .kaggle folder
New-Item -ItemType Directory -Path $env:USERPROFILE\.kaggle -Force

# Move kaggle.json there
Move-Item -Path .\Downloads\kaggle.json -Destination $env:USERPROFILE\.kaggle\kaggle.json -Force
```

**Linux/Mac:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Verify Setup

```powershell
kaggle datasets list
```

If you see a list of datasets, you're ready! ✅

---

## Workflow (On RTX 4060 Laptop)

### Step 1: Clone the repo and get existing datasets

```powershell
# Clone training repo
git clone https://github.com/amruthssss/yolo.git
cd yolo

# Copy your existing fighting/accident dataset
# Option A: Extract from zip you transferred
Expand-Archive -Path ..\yolo_dataset_fixed.zip -DestinationPath .\yolo_ready -Force

# Option B: Copy from USB/network
Copy-Item -Path E:\yolo_ready -Destination .\yolo_ready -Recurse
```

### Step 2: Download fire & explosion datasets from Kaggle

```powershell
python download_and_organize_kaggle_datasets.py --download-all --kaggle-dir kaggle_downloads
```

This will:
- Download fire datasets from Kaggle
- Download explosion datasets (or use alternatives if not available)
- Extract to `kaggle_downloads/fire/` and `kaggle_downloads/explosion/`

**Time:** 5-15 minutes depending on internet speed.

### Step 3: Organize into YOLO format

```powershell
python download_and_organize_kaggle_datasets.py --organize --yolo-root yolo_ready --max-samples 500
```

This will:
- Find all images in downloaded datasets
- Generate bounding box labels automatically using edge detection
- Assign correct class IDs (fire=0, explosion=3)
- Copy images to `yolo_ready/images/train` and `/val`
- Create labels in `yolo_ready/labels/train` and `/val`
- Merge with your existing fighting/accident data

**Time:** 2-5 minutes.

### Step 4: Create final data.yaml

```powershell
python download_and_organize_kaggle_datasets.py --finalize --yolo-root yolo_ready
```

Output:
```
📊 Dataset summary:
  ✅ fire: 450 instances
  ✅ vehicle_accident: 1234 instances
  ✅ fighting: 5678 instances
  ✅ explosion: 380 instances

✅ Created yolo_ready/data.yaml
```

### Step 5: Verify dataset

```powershell
python verify_dataset.py
```

### Step 6: Start training!

```powershell
.\start_training.ps1 -DataPath .\yolo_ready -Model yolov11m.pt -Epochs 100 -Batch 8 -Accumulate 2 -AutoCreate
```

---

## Alternative: Finding Better Datasets

The script uses default Kaggle datasets, but you can customize:

### Finding Fire Datasets

Search on Kaggle:
- https://www.kaggle.com/search?q=fire+detection
- https://www.kaggle.com/search?q=wildfire

Recommended:
- `phylake1337/fire-dataset` (default in script)
- `ritupande/fire-detection-from-cctv`

### Finding Explosion Datasets

**Note:** Explosion datasets are rare on Kaggle. Alternatives:

1. **YouTube Downloads** (with proper licenses):
   ```powershell
   # Install youtube-dl
   pip install yt-dlp
   
   # Download explosion compilation videos
   yt-dlp "https://youtube.com/watch?v=..." -o explosion_videos/%(title)s.%(ext)s
   ```

2. **Public CCTV Footage Repositories:**
   - Search GitHub for "explosion detection dataset"
   - Check academic dataset repositories (UCF-Crime, etc.)

3. **Manual Collection:**
   - News footage (check licensing)
   - Industrial safety videos

### Updating the Script with Custom Datasets

Edit `download_and_organize_kaggle_datasets.py`:

```python
KAGGLE_DATASETS = {
    'fire': [
        'phylake1337/fire-dataset',
        'your-username/your-fire-dataset',  # Add your custom dataset
    ],
    'explosion': [
        'another-username/explosion-dataset',
    ]
}
```

---

## Troubleshooting

### "401 Unauthorized" Error

Your kaggle.json is not in the correct location or has wrong permissions.

**Fix (Windows):**
```powershell
Move-Item -Path .\kaggle.json -Destination $env:USERPROFILE\.kaggle\kaggle.json -Force
```

**Fix (Linux/Mac):**
```bash
chmod 600 ~/.kaggle/kaggle.json
```

### "Dataset Not Found"

The Kaggle dataset ID might be wrong or private.

**Fix:**
1. Go to the dataset page on Kaggle
2. Copy the dataset ID from URL: `kaggle.com/datasets/<owner>/<dataset-name>`
3. Update `KAGGLE_DATASETS` in the script

### "No images found for fire/explosion"

The downloaded dataset might have a different structure.

**Fix:**
```powershell
# Manually inspect what was downloaded
Get-ChildItem -Recurse kaggle_downloads\fire

# If images are in a subfolder, move them
Move-Item kaggle_downloads\fire\subfolder\*.jpg kaggle_downloads\fire\
```

### "Bounding boxes look wrong"

The automatic edge detection might not work well for all images.

**Option 1:** Adjust detection in script (edit `detect_motion_bbox` function)

**Option 2:** Use manual labeling tool:
- Install labelImg: `pip install labelImg`
- Run: `labelImg yolo_ready/images/train yolo_ready/labels/train`

---

## Manual Alternative (If Kaggle Fails)

If Kaggle downloads don't work, use the video-based approach:

```powershell
# 1. Download videos manually from YouTube/other sources
# Place in fire_videos/ and explosion_videos/

# 2. Use the prepare_complete_dataset.py script
python prepare_complete_dataset.py --add-class fire --videos fire_videos
python prepare_complete_dataset.py --add-class explosion --videos explosion_videos
python prepare_complete_dataset.py --finalize
```

---

## Summary Checklist

- [ ] Install Kaggle API and setup credentials
- [ ] Clone GitHub repo on laptop
- [ ] Copy existing fighting/accident dataset to `yolo_ready/`
- [ ] Run `--download-all` to get fire & explosion from Kaggle
- [ ] Run `--organize` to merge into YOLO format
- [ ] Run `--finalize` to create data.yaml
- [ ] Verify with `verify_dataset.py`
- [ ] Start training with `start_training.ps1`

---

**Time estimate:** 15-30 minutes (mostly waiting for downloads)

**Disk space needed:** ~5-10 GB for all datasets combined
