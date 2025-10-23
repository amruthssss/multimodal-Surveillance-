#!/usr/bin/env python3
"""
prepare_complete_dataset.py

Helper script to prepare a complete 4-class YOLO dataset (fighting, vehicle_accident, fire, explosion).

Features:
- Scans existing yolo_ready/ to see what classes you have
- Helps you organize videos/images for missing classes
- Extracts frames from videos using motion detection
- Generates YOLO labels (bounding boxes covering detected motion regions)
- Merges into yolo_ready/ with train/val split
- Creates data.yaml with all 4 classes

Usage:
  python prepare_complete_dataset.py --scan                          # Check current dataset
  python prepare_complete_dataset.py --add-class fire --videos fire_videos/  # Add fire class
  python prepare_complete_dataset.py --finalize                      # Create data.yaml
"""

import argparse
import cv2
import random
import shutil
from pathlib import Path
from collections import defaultdict

# Class mapping (YOLO format: class_id is the index)
CLASS_NAMES = ['fire', 'vehicle_accident', 'fighting', 'explosion']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

VAL_RATIO = 0.2
MOTION_THRESHOLD = 25  # motion detection sensitivity
MIN_CONTOUR_AREA = 500  # minimum area for motion bounding box


def scan_dataset(root='datasets/yolo_ready'):
    """Scan existing dataset and report class distribution."""
    root = Path(root)
    if not root.exists():
        print(f"❌ Dataset folder not found: {root}")
        return {}
    
    print(f"📊 Scanning dataset at: {root.resolve()}")
    
    class_counts = defaultdict(int)
    for split in ['train', 'val']:
        labels_dir = root / 'labels' / split
        if not labels_dir.exists():
            print(f"⚠️ Labels folder not found: {labels_dir}")
            continue
        
        for label_file in labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if 0 <= class_id < len(CLASS_NAMES):
                                class_counts[CLASS_NAMES[class_id]] += 1
            except Exception as e:
                print(f"Error reading {label_file}: {e}")
    
    print("\n📈 Class distribution:")
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {class_name}: {count} instances")
    
    missing = [name for name in CLASS_NAMES if class_counts.get(name, 0) == 0]
    if missing:
        print(f"\n⚠️ Missing classes: {', '.join(missing)}")
        print("Use --add-class to add videos/images for these classes.")
    else:
        print("\n✅ All 4 classes present!")
    
    return class_counts


def extract_frames_from_video(video_path, output_dir, class_name, max_frames=100):
    """Extract frames with motion detection from a video."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return []
    
    print(f"📹 Processing video: {video_path.name}")
    
    # Motion detection setup
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=MOTION_THRESHOLD)
    
    frame_count = 0
    saved_count = 0
    prev_gray = None
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret or saved_count >= max_frames:
            break
        
        frame_count += 1
        
        # Skip some frames for efficiency
        if frame_count % 5 != 0:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = bg_subtractor.apply(gray)
        
        # Find contours (motion regions)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant motion
        significant_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
        
        if significant_contours:
            # Save frame
            frame_name = f"{class_name}_{video_path.stem}_frame_{saved_count:04d}.jpg"
            frame_path = output_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            
            # Generate YOLO label (bounding box around motion region)
            h, w = frame.shape[:2]
            boxes = []
            for contour in significant_contours:
                x, y, cw, ch = cv2.boundingRect(contour)
                # Convert to YOLO format (normalized center_x, center_y, width, height)
                center_x = (x + cw / 2) / w
                center_y = (y + ch / 2) / h
                norm_w = cw / w
                norm_h = ch / h
                boxes.append((center_x, center_y, norm_w, norm_h))
            
            extracted_frames.append((frame_path, boxes))
            saved_count += 1
        
        prev_gray = gray
    
    cap.release()
    print(f"  ✅ Extracted {saved_count} frames")
    return extracted_frames


def add_class_from_videos(class_name, videos_dir, dataset_root='datasets/yolo_ready', max_frames_per_video=100):
    """Add a class by processing videos and generating labels."""
    videos_dir = Path(videos_dir)
    dataset_root = Path(dataset_root)
    
    if class_name not in CLASS_MAP:
        print(f"❌ Invalid class name: {class_name}. Valid: {CLASS_NAMES}")
        return
    
    class_id = CLASS_MAP[class_name]
    print(f"\n🔥 Adding class '{class_name}' (id={class_id}) from videos in: {videos_dir}")
    
    if not videos_dir.exists():
        print(f"❌ Videos directory not found: {videos_dir}")
        return
    
    # Find video files
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(videos_dir.glob(ext))
    
    if not video_files:
        print(f"❌ No video files found in {videos_dir}")
        return
    
    print(f"Found {len(video_files)} video(s)")
    
    # Temporary extraction folder
    temp_dir = dataset_root / 'temp_extracted' / class_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    all_frames = []
    for video in video_files:
        frames = extract_frames_from_video(video, temp_dir, class_name, max_frames_per_video)
        all_frames.extend(frames)
    
    if not all_frames:
        print("⚠️ No frames extracted (no motion detected or videos empty)")
        return
    
    print(f"\n📦 Total frames extracted: {len(all_frames)}")
    
    # Split into train/val
    random.seed(42)
    random.shuffle(all_frames)
    n_val = max(1, int(len(all_frames) * VAL_RATIO))
    val_frames = all_frames[:n_val]
    train_frames = all_frames[n_val:]
    
    print(f"  Train: {len(train_frames)}, Val: {len(val_frames)}")
    
    # Move to yolo_ready structure
    for split, frames in [('train', train_frames), ('val', val_frames)]:
        img_dir = dataset_root / 'images' / split
        lbl_dir = dataset_root / 'labels' / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for frame_path, boxes in frames:
            # Copy image
            dest_img = img_dir / frame_path.name
            shutil.copy2(frame_path, dest_img)
            
            # Write label
            label_path = lbl_dir / (frame_path.stem + '.txt')
            with open(label_path, 'w') as f:
                for (cx, cy, w, h) in boxes:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"✅ Added {len(all_frames)} samples for class '{class_name}' to dataset")


def finalize_dataset(dataset_root='datasets/yolo_ready'):
    """Create final data.yaml with all 4 classes."""
    dataset_root = Path(dataset_root)
    
    train_path = dataset_root / 'images' / 'train'
    val_path = dataset_root / 'images' / 'val'
    
    if not train_path.exists() or not val_path.exists():
        print(f"❌ Missing images/train or images/val in {dataset_root}")
        return
    
    yaml_content = f"""train: {train_path.resolve()}
val: {val_path.resolve()}
nc: 4
names: {CLASS_NAMES}
"""
    
    yaml_path = dataset_root / 'data.yaml'
    yaml_path.write_text(yaml_content)
    
    print(f"\n✅ Created data.yaml at: {yaml_path}")
    print("\nContents:")
    print(yaml_content)
    
    # Final stats
    train_imgs = len(list(train_path.glob('*.jpg')))
    val_imgs = len(list(val_path.glob('*.jpg')))
    print(f"\n📊 Final dataset stats:")
    print(f"  Train images: {train_imgs}")
    print(f"  Val images: {val_imgs}")
    print(f"  Total: {train_imgs + val_imgs}")


def main():
    parser = argparse.ArgumentParser(description='Prepare complete 4-class YOLO dataset')
    parser.add_argument('--scan', action='store_true', help='Scan existing dataset')
    parser.add_argument('--add-class', type=str, help='Class name to add (fire, explosion, etc.)')
    parser.add_argument('--videos', type=str, help='Path to folder containing videos for the class')
    parser.add_argument('--finalize', action='store_true', help='Create data.yaml and finalize dataset')
    parser.add_argument('--dataset-root', default='datasets/yolo_ready', help='Root of yolo_ready dataset')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames per video')
    
    args = parser.parse_args()
    
    if args.scan:
        scan_dataset(args.dataset_root)
    elif args.add_class and args.videos:
        add_class_from_videos(args.add_class, args.videos, args.dataset_root, args.max_frames)
        print("\n💡 Run --scan to check updated distribution")
    elif args.finalize:
        finalize_dataset(args.dataset_root)
    else:
        parser.print_help()
        print("\n📝 Quick workflow:")
        print("  1. python prepare_complete_dataset.py --scan")
        print("  2. Place fire videos in fire_videos/ folder")
        print("  3. python prepare_complete_dataset.py --add-class fire --videos fire_videos/")
        print("  4. Place explosion videos in explosion_videos/ folder")
        print("  5. python prepare_complete_dataset.py --add-class explosion --videos explosion_videos/")
        print("  6. python prepare_complete_dataset.py --finalize")


if __name__ == '__main__':
    main()
