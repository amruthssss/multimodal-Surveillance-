#!/usr/bin/env python3
"""
download_and_organize_kaggle_datasets.py

Downloads fire and explosion datasets from Kaggle, organizes them into YOLO format,
and merges with existing fighting/accident datasets.

Prerequisites:
  1. Install Kaggle API: pip install kaggle
  2. Setup Kaggle credentials: https://www.kaggle.com/docs/api
     - Go to Kaggle → Account → Create New API Token
     - Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)

Usage:
  python download_and_organize_kaggle_datasets.py --download-all
  python download_and_organize_kaggle_datasets.py --organize
  python download_and_organize_kaggle_datasets.py --finalize
"""

import argparse
import json
import os
import random
import shutil
import zipfile
from pathlib import Path
import cv2
import numpy as np

# Kaggle dataset identifiers (update these with actual working datasets)
KAGGLE_DATASETS = {
    'fire': [
        'phylake1337/fire-dataset',  # Fire detection images
        'atulyakumar98/test-dataset',  # Alternative fire dataset
    ],
    'explosion': [
        'mhskjelvareid/facesinthewild',  # Placeholder - update with real explosion dataset
        # Note: Explosion datasets are rare on Kaggle. You may need to use YouTube downloads.
    ]
}

CLASS_NAMES = ['fire', 'vehicle_accident', 'fighting', 'explosion']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
VAL_RATIO = 0.2


def setup_kaggle():
    """Check if Kaggle API is configured."""
    try:
        import kaggle
        print("✅ Kaggle API is installed and configured")
        return True
    except OSError as e:
        print("❌ Kaggle API not configured properly")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section and click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in:")
        print("   Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("   Linux/Mac: ~/.kaggle/kaggle.json")
        print("5. Run: pip install kaggle")
        return False
    except ImportError:
        print("❌ Kaggle package not installed")
        print("Run: pip install kaggle")
        return False


def download_kaggle_dataset(dataset_id, download_dir):
    """Download a Kaggle dataset."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading {dataset_id}...")
    try:
        api.dataset_download_files(dataset_id, path=str(download_dir), unzip=True)
        print(f"✅ Downloaded and extracted to {download_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {dataset_id}: {e}")
        return False


def download_all_datasets(base_dir='kaggle_downloads'):
    """Download all fire and explosion datasets from Kaggle."""
    base_dir = Path(base_dir)
    
    if not setup_kaggle():
        return False
    
    downloaded = {}
    
    for class_name, dataset_ids in KAGGLE_DATASETS.items():
        print(f"\n🔥 Downloading {class_name} datasets...")
        class_dir = base_dir / class_name
        
        for dataset_id in dataset_ids:
            download_dir = class_dir / dataset_id.replace('/', '_')
            if download_kaggle_dataset(dataset_id, download_dir):
                if class_name not in downloaded:
                    downloaded[class_name] = []
                downloaded[class_name].append(download_dir)
    
    print(f"\n✅ Downloaded datasets for: {list(downloaded.keys())}")
    return downloaded


def find_images_in_dir(directory):
    """Recursively find all image files in a directory."""
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = []
    for ext in image_exts:
        images.extend(Path(directory).rglob(f'*{ext}'))
    return images


def create_simple_bbox(img_shape):
    """Create a simple bounding box covering most of the image (for datasets without bbox annotations)."""
    h, w = img_shape[:2]
    # Create a box covering 80% of image center
    margin = 0.1
    cx, cy = 0.5, 0.5
    bw, bh = 0.8, 0.8
    return (cx, cy, bw, bh)


def detect_motion_bbox(image_path):
    """Use edge detection to create a bounding box around the main object."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # No edges found, use default bbox
        return create_simple_bbox(img.shape)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Convert to YOLO format (normalized)
    img_h, img_w = img.shape[:2]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    
    return (cx, cy, nw, nh)


def organize_class_data(class_name, source_dirs, yolo_root='yolo_ready', max_samples=500):
    """Organize downloaded images into YOLO format."""
    yolo_root = Path(yolo_root)
    class_id = CLASS_MAP[class_name]
    
    print(f"\n📦 Organizing {class_name} (class_id={class_id})...")
    
    # Collect all images
    all_images = []
    for source_dir in source_dirs:
        images = find_images_in_dir(source_dir)
        all_images.extend(images)
    
    if not all_images:
        print(f"⚠️ No images found for {class_name}")
        return
    
    print(f"Found {len(all_images)} images")
    
    # Limit samples
    if len(all_images) > max_samples:
        random.seed(42)
        all_images = random.sample(all_images, max_samples)
        print(f"Limited to {max_samples} samples")
    
    # Split train/val
    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * VAL_RATIO))
    val_images = all_images[:n_val]
    train_images = all_images[n_val:]
    
    print(f"Train: {len(train_images)}, Val: {len(val_images)}")
    
    # Copy and create labels
    for split, images in [('train', train_images), ('val', val_images)]:
        img_dir = yolo_root / 'images' / split
        lbl_dir = yolo_root / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, img_path in enumerate(images):
            # Copy image with new name
            new_name = f"{class_name}_{idx:04d}{img_path.suffix}"
            dest_img = img_dir / new_name
            shutil.copy2(img_path, dest_img)
            
            # Create YOLO label
            bbox = detect_motion_bbox(img_path)
            if bbox is None:
                bbox = create_simple_bbox(cv2.imread(str(img_path)).shape)
            
            label_path = lbl_dir / (dest_img.stem + '.txt')
            with open(label_path, 'w') as f:
                cx, cy, w, h = bbox
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    print(f"✅ Organized {len(all_images)} samples for {class_name}")


def organize_all(kaggle_dir='kaggle_downloads', yolo_root='yolo_ready'):
    """Organize all downloaded Kaggle datasets into YOLO format."""
    kaggle_dir = Path(kaggle_dir)
    
    if not kaggle_dir.exists():
        print(f"❌ Kaggle downloads directory not found: {kaggle_dir}")
        print("Run with --download-all first")
        return
    
    for class_name in ['fire', 'explosion']:
        class_dir = kaggle_dir / class_name
        if not class_dir.exists():
            print(f"⚠️ No {class_name} downloads found")
            continue
        
        # Find all subdirs (one per dataset)
        source_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        if source_dirs:
            organize_class_data(class_name, source_dirs, yolo_root)


def finalize_dataset(yolo_root='yolo_ready'):
    """Create final data.yaml with all 4 classes."""
    yolo_root = Path(yolo_root)
    
    train_path = yolo_root / 'images' / 'train'
    val_path = yolo_root / 'images' / 'val'
    
    if not train_path.exists() or not val_path.exists():
        print(f"❌ Missing images folders in {yolo_root}")
        return
    
    # Count samples per class
    print("\n📊 Dataset summary:")
    class_counts = {name: 0 for name in CLASS_NAMES}
    
    for split in ['train', 'val']:
        lbl_dir = yolo_root / 'labels' / split
        if lbl_dir.exists():
            for lbl_file in lbl_dir.glob('*.txt'):
                with open(lbl_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cid = int(parts[0])
                            if 0 <= cid < len(CLASS_NAMES):
                                class_counts[CLASS_NAMES[cid]] += 1
    
    for class_name in CLASS_NAMES:
        status = "✅" if class_counts[class_name] > 0 else "❌"
        print(f"  {status} {class_name}: {class_counts[class_name]} instances")
    
    # Create data.yaml
    yaml_content = f"""train: {train_path.resolve()}
val: {val_path.resolve()}
nc: 4
names: {CLASS_NAMES}
"""
    
    yaml_path = yolo_root / 'data.yaml'
    yaml_path.write_text(yaml_content)
    
    print(f"\n✅ Created {yaml_path}")
    print("\n" + yaml_content)
    
    train_imgs = len(list(train_path.glob('*.jpg'))) + len(list(train_path.glob('*.png')))
    val_imgs = len(list(val_path.glob('*.jpg'))) + len(list(val_path.glob('*.png')))
    print(f"Total images: Train={train_imgs}, Val={val_imgs}")


def main():
    parser = argparse.ArgumentParser(description='Download and organize Kaggle datasets for YOLO training')
    parser.add_argument('--download-all', action='store_true', help='Download fire and explosion datasets from Kaggle')
    parser.add_argument('--organize', action='store_true', help='Organize downloaded datasets into YOLO format')
    parser.add_argument('--finalize', action='store_true', help='Create data.yaml and finalize dataset')
    parser.add_argument('--kaggle-dir', default='kaggle_downloads', help='Directory for Kaggle downloads')
    parser.add_argument('--yolo-root', default='yolo_ready', help='Root directory for YOLO dataset')
    parser.add_argument('--max-samples', type=int, default=500, help='Max samples per class')
    
    args = parser.parse_args()
    
    if args.download_all:
        download_all_datasets(args.kaggle_dir)
    elif args.organize:
        organize_all(args.kaggle_dir, args.yolo_root)
    elif args.finalize:
        finalize_dataset(args.yolo_root)
    else:
        parser.print_help()
        print("\n📝 Complete workflow:")
        print("  1. Setup Kaggle API credentials (see instructions above)")
        print("  2. python download_and_organize_kaggle_datasets.py --download-all")
        print("  3. python download_and_organize_kaggle_datasets.py --organize")
        print("  4. python download_and_organize_kaggle_datasets.py --finalize")
        print("\nThis will create a complete yolo_ready/ folder with all 4 classes!")


if __name__ == '__main__':
    main()
