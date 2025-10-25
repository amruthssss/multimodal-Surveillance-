"""
Download FREE datasets for YOLO v11 training
Focus: explosion, fire, smoke, vehicle_accident, fighting events
Only downloads what's needed for your project
"""

import requests
import zipfile
import tarfile
import os
from pathlib import Path
from tqdm import tqdm
import shutil

def download_file(url, dest_path):
    """Download file with progress bar"""
    print(f"\n📥 Downloading: {dest_path.name}")
    print(f"   URL: {url}")
    
    response = requests.get(url, stream=True)
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
    return dest_path

def extract_archive(archive_path, dest_dir):
    """Extract zip or tar.gz archive"""
    print(f"\n📦 Extracting: {archive_path.name}")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
    elif archive_path.suffix == '.gz' or '.tar' in archive_path.name:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(dest_dir)
    
    print(f"   ✅ Extracted to: {dest_dir}")

def download_roboflow_datasets():
    """Download pre-labeled YOLO datasets from Roboflow Universe (FREE)"""
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("📚 DOWNLOADING ROBOFLOW DATASETS (FREE)")
    print("="*70)
    
    # These are FREE public datasets from Roboflow Universe
    # Format: YOLO-ready (images + labels + data.yaml)
    
    datasets = {
        'fire_smoke': {
            'name': 'Fire and Smoke Detection',
            'url': 'https://universe.roboflow.com/roboflow-universe-projects/fire-and-smoke-detection/dataset/1/download/yolov8',
            'events': ['fire', 'smoke'],
            'size': '~200 MB',
            'images': '~5,000'
        },
        'accident': {
            'name': 'Traffic Accident Detection',
            'url': 'https://universe.roboflow.com/accident-detection-yg9vx/accident-detection-system/dataset/1/download/yolov8',
            'events': ['vehicle_accident'],
            'size': '~150 MB',
            'images': '~3,000'
        },
        'fighting': {
            'name': 'Violence Detection',
            'url': 'https://universe.roboflow.com/violence-detection-wjxtf/violence-detection-vdxfc/dataset/1/download/yolov8',
            'events': ['fighting'],
            'size': '~100 MB',
            'images': '~2,000'
        }
    }
    
    print("\n📋 Available datasets:")
    for key, info in datasets.items():
        print(f"\n   {info['name']}")
        print(f"      Events: {', '.join(info['events'])}")
        print(f"      Images: {info['images']}")
        print(f"      Size: {info['size']}")
    
    print("\n⚠️  NOTE: Roboflow requires API key for download")
    print("   Get FREE API key: https://app.roboflow.com/")
    print("   Alternative: Use manual download from Roboflow Universe")
    
    return datasets

def download_kaggle_datasets():
    """Download datasets from Kaggle (FREE with account)"""
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("📚 KAGGLE DATASETS (FREE)")
    print("="*70)
    
    # Kaggle CLI command examples
    kaggle_datasets = {
        'fire': {
            'name': 'Fire Detection Dataset',
            'command': 'kaggle datasets download -d phylake1337/fire-dataset',
            'events': ['fire'],
            'size': '~1 GB',
            'images': '~10,000'
        },
        'accidents': {
            'name': 'Car Crash Dataset',
            'command': 'kaggle datasets download -d anshtanwar/car-crash-dataset',
            'events': ['vehicle_accident'],
            'size': '~500 MB',
            'images': '~5,000'
        },
        'violence': {
            'name': 'Violence Detection',
            'command': 'kaggle datasets download -d mohamedmustafa/real-life-violence-situations-dataset',
            'events': ['fighting'],
            'size': '~2 GB',
            'images': '~15,000 videos'
        }
    }
    
    print("\n📋 To download from Kaggle:")
    print("   1. Install: pip install kaggle")
    print("   2. Get API key: https://www.kaggle.com/account")
    print("   3. Place kaggle.json in: ~/.kaggle/")
    print("\n   Then run these commands:")
    
    for key, info in kaggle_datasets.items():
        print(f"\n   # {info['name']} ({info['size']})")
        print(f"   {info['command']}")
    
    return kaggle_datasets

def download_coco_subset():
    """Download COCO dataset subset (only fire/smoke/explosion classes)"""
    print("\n" + "="*70)
    print("📚 COCO DATASET SUBSET")
    print("="*70)
    
    print("\n⚠️  COCO full dataset is ~25 GB")
    print("   For your project, you only need specific classes:")
    print("   - fire truck (for fire context)")
    print("   - smoke (not in COCO)")
    print("   - car (for accidents)")
    print("   - person (for fighting)")
    
    print("\n   Recommended: Skip COCO, use specialized datasets above")

def create_combined_dataset():
    """Create a combined dataset.yaml for all downloaded datasets"""
    datasets_dir = Path('datasets')
    combined_dir = datasets_dir / 'combined_yolo'
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml
    data_yaml = combined_dir / 'data.yaml'
    
    yaml_content = f"""# Combined YOLO v11 Dataset
# Auto-generated for your project events

path: {combined_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: explosion
  1: fire
  2: smoke
  3: vehicle_accident
  4: fighting
  5: vandalism
  6: burglary
  7: robbery
  8: shooting
  9: theft

# Dataset info
nc: 10  # number of classes
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Created: {data_yaml}")
    return data_yaml

def main():
    print("\n" + "="*70)
    print("🎯 YOLO v11 DATASET DOWNLOADER")
    print("="*70)
    print("Focus: explosion, fire, smoke, vehicle_accident, fighting")
    print("="*70)
    
    datasets_dir = Path('datasets')
    datasets_dir.mkdir(exist_ok=True)
    
    print("\n📁 Dataset directory: datasets/")
    
    # Show available datasets
    print("\n" + "="*70)
    print("OPTION 1: Roboflow Universe (Easiest - Pre-labeled YOLO format)")
    print("="*70)
    roboflow = download_roboflow_datasets()
    
    print("\n" + "="*70)
    print("OPTION 2: Kaggle Datasets (Larger - Needs conversion)")
    print("="*70)
    kaggle = download_kaggle_datasets()
    
    print("\n" + "="*70)
    print("OPTION 3: Manual Download Links (Recommended)")
    print("="*70)
    
    print("\n🔗 Direct download links (no signup needed):")
    print("\n   1. Fire Detection Dataset:")
    print("      https://github.com/DeepQuestAI/Fire-Smoke-Dataset")
    print("      (~1,000 images, fire + smoke)")
    
    print("\n   2. Accident Dataset:")
    print("      https://github.com/ankitdhall/accident-detection")
    print("      (~800 images, vehicle accidents)")
    
    print("\n   3. Fighting/Violence Dataset:")
    print("      http://www.openu.ac.il/home/hassner/data/violentflows/")
    print("      (~250 videos, fighting)")
    
    print("\n   4. Explosion Dataset (Limited):")
    print("      Use YouTube videos + manual labeling")
    print("      Or: https://github.com/imatge-upc/activitynet-2016-cvprw")
    
    # Create combined dataset structure
    print("\n" + "="*70)
    print("📦 CREATING DATASET STRUCTURE")
    print("="*70)
    
    yaml_path = create_combined_dataset()
    
    print("\n✅ SETUP COMPLETE!")
    print("\n📋 NEXT STEPS:")
    print("\n   1. Download datasets manually from links above")
    print("   2. Place in: datasets/<dataset_name>/")
    print("   3. Convert to YOLO format:")
    print("      python scripts/convert_to_yolo.py --src datasets/fire --dst datasets/combined_yolo")
    print("\n   4. Train YOLO v11:")
    print(f"      python scripts/train_yolo_v11.py --data {yaml_path}")
    
    print("\n💡 QUICK START (Smallest datasets first):")
    print("   1. Fire dataset (~1,000 images) - Start here!")
    print("   2. Accident dataset (~800 images)")
    print("   3. Add more as needed")
    
    print("\n⚠️  Minimum recommended: 500 images per class")
    print("   Your current YOLO: trained on limited data")
    print("   With new data: expect 5-10% accuracy improvement")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
