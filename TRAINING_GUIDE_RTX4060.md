# YOLOv11 Training Guide for RTX 4060 Laptop

## 📦 Dataset Ready
- **File:** `yolo_dataset_final.zip` (684.5 MB)
- **Total Images:** 3,182 high-quality event frames
  - Train: 2,546 images
  - Val: 636 images
- **Classes:** 4 (fire, vehicle_accident, fighting, explosion)
- **Balance:** ~800 samples per class (fire: 782, others: 800)

## 🚀 Setup on RTX 4060 Laptop

### 1. Transfer Dataset
Copy `yolo_dataset_final.zip` to `D:\yolo\` on the RTX 4060 laptop via:
- USB drive (fastest for 684 MB)
- Network share
- OneDrive/Google Drive

### 2. Extract Dataset
```powershell
cd D:\yolo
Expand-Archive -Path yolo_dataset_final.zip -DestinationPath .
```

This creates: `D:\yolo\yolo_ready_final\`
```
yolo_ready_final/
├── images/
│   ├── train/  (2,546 images)
│   └── val/    (636 images)
├── labels/
│   ├── train/  (2,546 .txt files)
│   └── val/    (636 .txt files)
└── data.yaml
```

### 3. Verify Dataset Structure
```powershell
python verify_dataset.py --data yolo_ready_final
```

### 4. Start Training

#### Option A: Using PowerShell Helper (Recommended)
```powershell
.\start_training.ps1 -DataPath .\yolo_ready_final -Model yolov11m.pt -Epochs 150 -Batch 8 -Accumulate 4
```

#### Option B: Direct Python Command
```powershell
python train_yolov11.py --data yolo_ready_final --model yolov11m.pt --epochs 150 --batch 8 --accumulate 4 --imgsz 640 --auto-create --export-onnx
```

## ⚙️ Training Parameters Explained

### Optimized for RTX 4060 (8GB VRAM)
- **Model:** `yolov11m.pt` (medium - best balance accuracy/speed)
- **Batch Size:** 8 (fits in 8GB VRAM)
- **Gradient Accumulation:** 4 (effective batch = 32)
- **Epochs:** 150 (with early stopping patience=30)
- **Image Size:** 640x640
- **Optimizer:** AdamW (better generalization)
- **Learning Rate:** 0.002 → 0.001 (cosine decay)

### Key Hyperparameters for 95%+ mAP
```python
box=8.0         # High box weight for precise localization
cls=1.0         # High cls weight for 4-class distinction
patience=30     # Early stopping after 30 epochs without improvement
cache='disk'    # Cache dataset to disk (3K+ images)
rect=True       # Rectangular training for faster convergence
mosaic=1.0      # Multi-scale learning
mixup=0.1       # Light regularization
```

## 📊 Expected Training Time
- **RTX 4060:** ~2-3 hours for 150 epochs
- **Early stopping:** May finish earlier if converged (patience=30)

## 🎯 Target Metrics (95%+ mAP)
```
Metric          Target
─────────────────────────
mAP@0.5         > 0.95
mAP@0.5:0.95    > 0.75
Precision       > 0.90
Recall          > 0.90
```

## 📁 Training Outputs
Results will be saved to:
```
D:\yolo\runs\yolo_cctv_4class\
├── weights/
│   ├── best.pt      # Best model (use for inference)
│   ├── last.pt      # Latest checkpoint
│   └── best.onnx    # Exported ONNX (for deployment)
├── results.csv      # Metrics per epoch
├── results.png      # Training curves
├── confusion_matrix.png
└── PR_curve.png     # Precision-Recall curves
```

## 🔍 Monitor Training

### Real-time Metrics
Watch the terminal output:
```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  mAP@0.5  mAP@0.5:0.95
  1/150   5.2G    1.234     0.567     1.123      100      0.452      0.321
  2/150   5.3G    1.156     0.523     1.089      100      0.523      0.378
  ...
 50/150   5.4G    0.456     0.234     0.567      100      0.892      0.712
 ...
100/150   5.4G    0.234     0.123     0.345      100      0.956      0.782  ⭐ Best
```

### Check Results
```powershell
# View training curves
ii runs\yolo_cctv_4class\results.png

# View confusion matrix
ii runs\yolo_cctv_4class\confusion_matrix.png

# Check metrics CSV
Import-Csv runs\yolo_cctv_4class\results.csv | Format-Table
```

## 🛠️ Troubleshooting

### Out of Memory (OOM)
Reduce batch size:
```powershell
.\start_training.ps1 -Batch 4 -Accumulate 8  # Same effective batch, less VRAM
```

### Training Too Slow
Use smaller model:
```powershell
.\start_training.ps1 -Model yolov11n.pt  # nano model (faster, slightly less accurate)
```

### Training Not Converging
Increase epochs or adjust LR:
```powershell
.\start_training.ps1 -Epochs 200  # More epochs for complex patterns
```

### Resume Training (if interrupted)
```powershell
python train_yolov11.py --data yolo_ready_final --resume runs/yolo_cctv_4class/weights/last.pt
```

## 📈 After Training

### Test the Model
```python
from ultralytics import YOLO

model = YOLO('runs/yolo_cctv_4class/weights/best.pt')
results = model.predict('test_video.mp4', save=True)
```

### Export for Deployment
```python
model = YOLO('runs/yolo_cctv_4class/weights/best.pt')
model.export(format='onnx')     # For C++/production
model.export(format='tflite')   # For mobile/edge devices
model.export(format='engine')   # For TensorRT (fastest inference)
```

## 🎓 What Makes This Dataset Optimized for 95%+ mAP

1. ✅ **Quality Over Quantity**
   - Only event frames extracted (not all video frames)
   - Motion detection removes static/boring frames
   - Blur detection removes low-quality frames
   - Duplicate removal prevents overfitting

2. ✅ **Balanced Classes**
   - Each class has ~800 samples
   - Prevents model bias toward dominant classes
   - Better multi-class discrimination

3. ✅ **Proper Train/Val Split**
   - 80/20 split ensures good validation
   - Random shuffle prevents temporal bias

4. ✅ **YOLO-Format Labels**
   - Bounding boxes for each object
   - Normalized coordinates (0-1)
   - Class IDs: 0=fire, 1=vehicle_accident, 2=fighting, 3=explosion

5. ✅ **Optimized Hyperparameters**
   - AdamW optimizer for generalization
   - High box/cls loss weights for CCTV
   - Rectangular training for faster convergence
   - Smart augmentation (no flipud, low perspective)

## 🚀 Quick Start Command
```powershell
# All-in-one command (copy-paste ready)
cd D:\yolo; Expand-Archive yolo_dataset_final.zip .; .\start_training.ps1 -DataPath .\yolo_ready_final -Epochs 150
```

---
**Created:** October 23, 2025  
**Dataset:** yolo_dataset_final.zip (684.5 MB, 3,182 images)  
**Target:** 95%+ mAP on 4-class CCTV event detection
