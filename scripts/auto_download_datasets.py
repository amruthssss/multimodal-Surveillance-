"""
AUTO-DOWNLOAD FREE DATASETS FOR YOLO v11
Downloads datasets that don't require API keys
Focus: fire, smoke, accident, fighting
"""

import requests
import os
from pathlib import Path
from tqdm import tqdm
import zipfile
import shutil

def download_with_progress(url, dest_path):
    """Download file with progress bar"""
    print(f"\n📥 Downloading: {dest_path.name}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"   ✅ Downloaded: {dest_path}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def extract_zip(zip_path, dest_dir):
    """Extract zip file"""
    print(f"\n📦 Extracting: {zip_path.name}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"   ✅ Extracted to: {dest_dir}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to extract: {e}")
        return False

def download_fire_dataset():
    """Download Fire-Smoke dataset from GitHub"""
    print("\n" + "="*70)
    print("🔥 DOWNLOADING FIRE & SMOKE DATASET")
    print("="*70)
    
    datasets_dir = Path('datasets/fire_smoke')
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # GitHub repository with fire/smoke images
    urls = {
        'train': 'https://github.com/DeepQuestAI/Fire-Smoke-Dataset/archive/refs/heads/master.zip'
    }
    
    for name, url in urls.items():
        dest_path = datasets_dir / f'{name}.zip'
        if download_with_progress(url, dest_path):
            extract_zip(dest_path, datasets_dir)
            # Clean up zip
            dest_path.unlink()
    
    print(f"\n✅ Fire & Smoke dataset ready: {datasets_dir}")
    return datasets_dir

def download_sample_accidents():
    """Download sample accident images"""
    print("\n" + "="*70)
    print("🚗 DOWNLOADING ACCIDENT DATASET SAMPLES")
    print("="*70)
    
    datasets_dir = Path('datasets/accidents')
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # These are public domain accident images
    sample_urls = [
        'https://live.staticflickr.com/65535/48969171832_7a62eef86e_b.jpg',
        'https://live.staticflickr.com/1928/44937873144_c6b8b6e06e_b.jpg',
        'https://live.staticflickr.com/7817/47215231861_8e8fbbf858_b.jpg',
    ]
    
    print("⚠️  Note: Downloading sample images only")
    print("   For full dataset, use manual download from links above")
    
    for i, url in enumerate(sample_urls):
        dest_path = datasets_dir / f'accident_{i+1}.jpg'
        download_with_progress(url, dest_path)
    
    print(f"\n✅ Sample accidents ready: {datasets_dir}")
    return datasets_dir

def create_instructions():
    """Create instruction file"""
    instructions = """
YOLO v11 DATASET DOWNLOAD - COMPLETE GUIDE
==========================================

✅ WHAT WAS AUTO-DOWNLOADED:
- Fire & Smoke dataset (GitHub)
- Sample accident images

❌ WHAT NEEDS MANUAL DOWNLOAD:

1. KAGGLE DATASETS (Best Quality):
   ----------------------------------
   a) Install Kaggle CLI:
      pip install kaggle
   
   b) Get API key:
      - Go to: https://www.kaggle.com/account
      - Click "Create New API Token"
      - Save kaggle.json to: C:\\Users\\<YourName>\\.kaggle\\
   
   c) Download datasets:
      kaggle datasets download -d phylake1337/fire-dataset
      kaggle datasets download -d anshtanwar/car-crash-dataset
      kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset

2. ROBOFLOW DATASETS (Pre-labeled YOLO):
   --------------------------------------
   a) Sign up (FREE): https://app.roboflow.com/
   
   b) Browse datasets:
      - Fire & Smoke: https://universe.roboflow.com/search?q=fire
      - Accidents: https://universe.roboflow.com/search?q=accident
      - Fighting: https://universe.roboflow.com/search?q=violence
   
   c) Click "Download" → Select "YOLO v8" format
   
   d) Extract to: datasets/<dataset_name>/

3. GITHUB DATASETS (Free, No Signup):
   -----------------------------------
   a) Fire-Smoke Dataset:
      git clone https://github.com/DeepQuestAI/Fire-Smoke-Dataset
      mv Fire-Smoke-Dataset datasets/fire_smoke/
   
   b) Accident Detection:
      git clone https://github.com/ankitdhall/accident-detection
      mv accident-detection datasets/accidents/

NEXT STEPS:
===========

1. After downloading, organize datasets:
   datasets/
   ├── fire_smoke/
   │   ├── images/
   │   └── labels/  (if YOLO format)
   ├── accidents/
   └── fighting/

2. Convert to YOLO format (if needed):
   python scripts/convert_to_yolo.py --src datasets/fire_smoke --dst datasets/combined_yolo

3. Train YOLO v11:
   python scripts/train_yolo_v11.py --data datasets/combined_yolo/data.yaml --epochs 50

MINIMUM REQUIREMENTS:
====================
- 500 images per class (fire, smoke, accident, fighting)
- YOLO format: images/ + labels/ folders
- data.yaml with class names

ESTIMATED DOWNLOAD SIZES:
=========================
- Fire dataset: ~1 GB
- Accident dataset: ~500 MB
- Fighting dataset: ~2 GB
- Total: ~3.5 GB

TRAINING TIME:
==============
- 50 epochs on CPU: ~8-12 hours
- 50 epochs on GPU: ~1-2 hours
- Recommended: Use Google Colab (FREE GPU)

GOOGLE COLAB TRAINING:
=====================
1. Upload datasets to Google Drive
2. Open Colab: https://colab.research.google.com/
3. Mount Drive: from google.colab import drive; drive.mount('/content/drive')
4. Install ultralytics: !pip install ultralytics
5. Train: !yolo train data=/content/drive/MyDrive/datasets/data.yaml model=yolov11n.pt epochs=50

Questions? Check TRAINING_README.md
"""
    
    with open('DATASET_DOWNLOAD_GUIDE.md', 'w') as f:
        f.write(instructions)
    
    print(f"\n✅ Created: DATASET_DOWNLOAD_GUIDE.md")

def main():
    print("\n" + "="*70)
    print("🎯 AUTO-DOWNLOADING FREE DATASETS")
    print("="*70)
    print("Focus: fire, smoke, accidents (no API key needed)")
    print("="*70)
    
    # Install required packages
    print("\n📦 Checking dependencies...")
    try:
        import requests
        import tqdm
        print("   ✅ All dependencies installed")
    except ImportError:
        print("   ⚠️  Installing required packages...")
        os.system("pip install requests tqdm -q")
    
    # Create datasets directory
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    print(f"\n📁 Dataset directory: {datasets_dir.absolute()}")
    
    # Download what we can automatically
    success = []
    
    try:
        fire_dir = download_fire_dataset()
        success.append(('Fire & Smoke', fire_dir))
    except Exception as e:
        print(f"\n❌ Fire dataset failed: {e}")
    
    try:
        accident_dir = download_sample_accidents()
        success.append(('Accident Samples', accident_dir))
    except Exception as e:
        print(f"\n❌ Accident samples failed: {e}")
    
    # Create instruction guide
    create_instructions()
    
    # Summary
    print("\n" + "="*70)
    print("✅ AUTO-DOWNLOAD COMPLETE")
    print("="*70)
    
    if success:
        print("\n📥 Successfully downloaded:")
        for name, path in success:
            print(f"   ✅ {name}: {path}")
    
    print("\n📋 NEXT STEPS:")
    print("\n   1. Read: DATASET_DOWNLOAD_GUIDE.md")
    print("   2. Download more datasets using Kaggle/Roboflow (see guide)")
    print("   3. Convert to YOLO format:")
    print("      python scripts/convert_to_yolo.py --src datasets/fire_smoke --dst datasets/combined_yolo")
    print("   4. Train:")
    print("      python scripts/train_yolo_v11.py --data datasets/combined_yolo/data.yaml")
    
    print("\n💡 RECOMMENDED: Use Google Colab for FREE GPU training!")
    print("   Guide in: DATASET_DOWNLOAD_GUIDE.md")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
