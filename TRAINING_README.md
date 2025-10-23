# Training package for YOLOv11 (for another laptop)

This folder contains the files and instructions you need to run YOLOv11 training on another machine (e.g. a laptop with RTX 4060).

What to include in your GitHub push
- This repository should include the training scripts and environment files (these files).
- DO NOT add large dataset zips (e.g. `yolo_dataset_fixed.zip`) to Git — instead use Git LFS or place the dataset on external storage (Drive/HTTP). See notes below.

Files added
- `train_yolov11.py`  — training wrapper (already in repo root)
- `environment.yml`  — conda environment specification
- `start_training.ps1` — PowerShell helper to create env and run training on Windows
- `data_yaml.template` — template for `data.yaml` you should place in `yolo_ready/`
- `verify_dataset.py` — quick sanity checker for dataset layout
- `.gitattributes` — suggests Git LFS rules for large archives

Quick steps (on the target laptop)
1. Clone the repo:
   git clone <your-repo-url>
   cd <repo>

2. Place dataset on the laptop:
   - Recommended: copy the extracted `yolo_ready/` folder into the repo root (so `./yolo_ready/` exists), or
   - If you have a zip `yolo_dataset_fixed.zip`, extract it so it produces `yolo_ready/`.

3. Edit `yolo_ready/data.yaml` (use `data_yaml.template` as reference) and ensure `nc` and `names` match your classes.

4. Create the conda env and install dependencies (PowerShell):
   Open PowerShell and run:
   ```powershell
   conda env create -f environment.yml
   conda activate yolov11-train
   # Then install the CUDA-enabled PyTorch matching your driver, e.g. for CUDA 12.1:
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
   pip install -r requirements_extra.txt || pip install ultralytics tqdm pandas matplotlib
   ```

5. Run training using PowerShell starter (example):
   ```powershell
   # from repo root
   .\start_training.ps1 -DataPath .\yolo_ready -Model yolov11m.pt -Epochs 100 -Batch 8 -ImgSz 640 -Project .\runs -Name yolo_v11_95_percent
   ```

Notes
- If your dataset is large, consider using Git LFS for `.zip` or `.pt` files. Add them to `.gitattributes` and configure LFS before pushing.
- If you prefer `conda` commands instead of the environment file, see the comments inside `start_training.ps1`.

If you want, I can create a GitHub release (instructions) or prepare an upload script to copy the dataset to a remote URL for the other laptop to download.
# YOLO v11 Training Guide (for your project)

Goal: Train YOLO v11 on event datasets to improve detection accuracy for: explosion, fire, smoke, vehicle_accident (and optionally other events).

Prerequisites
- Python 3.8+ (your repo already uses Python)
- GPU + CUDA (recommended). CPU-only will be very slow.
- Install dependencies:

```powershell
pip install -r requirements.txt
# If ultralytics not in requirements:
pip install ultralytics
```

Prepare datasets
1. Collect labeled images for these events (recommended free datasets):
   - UCF Crime (acts like anomalies)
   - VIRAT
   - CAVIAR
   - Custom images from your expanded patterns (convert carefully)

2. Structure datasets as class folders: `dataset/<event_name>/*.jpg`
3. Create `events.txt` with one event label per line (order will determine class ids):
```
explosion
fire
smoke
vehicle_accident
fighting
vandalism
burglary
robbery
shooting
theft
```

4. Convert to YOLO dataset format (images + labels + data.yaml):

```powershell
python scripts/convert_to_yolo.py --src dataset --dst data/yolo_dataset --labels events.txt
```

Train YOLO v11

```powershell
python scripts/train_yolo_v11.py --data data/yolo_dataset/data.yaml --epochs 50 --batch 16 --img 640
```

Tips
- Start with small epochs (10-20) and inspect results.
- Use augmentation (albumentations) for more robustness.
- If you have imbalanced classes, use class weights or oversample minority classes.
- Use `--device 0` for GPU, or `--device cpu` for CPU.

Evaluate
- After training, check `runs/train/<name>/results.csv` and `best.pt`.
- Use `ultralytics` validation: `python -c "from ultralytics import YOLO; YOLO('runs/train/exp/weights/best.pt').val(data='data/yolo_dataset/data.yaml')"`

Deploy
- Replace `models/best.pt` with new `best.pt` or update `process_*` scripts to point to the new weights.

Notes
- If you want, I can: generate a small script to auto-download suggested datasets (where available), or create augmentation config and training hyperparameters tuned for your data.
