"""
Prepare downloaded datasets for YOLO v11 training
- Extracts frames from violence videos
- Organizes images into YOLO format
- Creates data.yaml
"""

import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

def detect_event_segments(video_path, motion_threshold=25, scene_change_threshold=30):
    """
    Detect segments where events occur using motion and scene change detection
    Returns list of (start_frame, end_frame) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    
    prev_frame = None
    event_segments = []
    current_segment_start = None
    motion_scores = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is not None:
            # Calculate frame difference (motion)
            frame_diff = cv2.absdiff(prev_frame, gray)
            motion_score = frame_diff.mean()
            
            # Detect event (high motion or scene change)
            if motion_score > motion_threshold:
                if current_segment_start is None:
                    current_segment_start = frame_idx
            else:
                # Low motion - end of event segment
                if current_segment_start is not None:
                    event_segments.append((current_segment_start, frame_idx))
                    current_segment_start = None
        
        prev_frame = gray.copy()
        frame_idx += 1
    
    # Handle last segment
    if current_segment_start is not None:
        event_segments.append((current_segment_start, frame_idx))
    
    cap.release()
    
    # Merge close segments (within 30 frames)
    if event_segments:
        merged_segments = [event_segments[0]]
        for start, end in event_segments[1:]:
            last_start, last_end = merged_segments[-1]
            if start - last_end < 30:  # Close segments
                merged_segments[-1] = (last_start, end)
            else:
                merged_segments.append((start, end))
        return merged_segments
    
    return []

def extract_frames_from_videos(video_dir, output_dir, frames_per_segment=10):
    """
    Extract frames from video files ONLY during event segments
    Uses motion detection to find where events occur
    """
    print(f"\n{'='*70}")
    print("🎬 EXTRACTING FRAMES FROM EVENT SEGMENTS (SMART DETECTION)")
    print(f"{'='*70}")
    
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all videos
    video_files = list(video_dir.rglob('*.avi')) + list(video_dir.rglob('*.mp4'))
    
    print(f"📹 Found {len(video_files)} videos")
    print(f"🎯 Detecting event segments with motion analysis...")
    print(f"📸 Extracting {frames_per_segment} frames per event segment")
    
    total_frames = 0
    total_segments = 0
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        # Detect event segments
        event_segments = detect_event_segments(video_path)
        
        if not event_segments:
            # No events detected, extract a few random frames
            cap = cv2.VideoCapture(str(video_path))
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if total_video_frames > 0:
                event_segments = [(0, min(total_video_frames, 30))]  # First 30 frames
        
        # Extract frames from each segment
        cap = cv2.VideoCapture(str(video_path))
        
        for segment_idx, (start_frame, end_frame) in enumerate(event_segments):
            segment_length = end_frame - start_frame
            interval = max(1, segment_length // frames_per_segment)
            
            for frame_idx in range(start_frame, end_frame, interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    output_path = output_dir / f"{video_path.stem}_seg{segment_idx:02d}_frame_{frame_idx:04d}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    total_frames += 1
                
                if total_frames % 100 == 0:
                    break  # Limit frames per segment
        
        cap.release()
        total_segments += len(event_segments)
    
    print(f"\n✅ Detected {total_segments} event segments")
    print(f"✅ Extracted {total_frames:,} frames from events only")
    print(f"💡 Skipped static/non-event portions of videos")
    return total_frames

def organize_yolo_dataset():
    """Organize datasets into YOLO format"""
    print(f"\n{'='*70}")
    print("📁 ORGANIZING YOLO DATASET")
    print(f"{'='*70}")
    
    # Create YOLO structure
    yolo_dir = Path('datasets/yolo_ready')
    yolo_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories
    for split in ['train', 'val']:
        (yolo_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    datasets_config = [
        {
            'source': 'datasets/fire',
            'event': 'fire',
            'class_id': 0
        },
        {
            'source': 'datasets/accidents_cctv',
            'event': 'vehicle_accident',
            'class_id': 1
        },
        {
            'source': 'datasets/violence_frames',
            'event': 'fighting',
            'class_id': 2
        }
    ]
    
    total_train = 0
    total_val = 0
    
    for config in datasets_config:
        source_dir = Path(config['source'])
        if not source_dir.exists():
            print(f"⚠️  Skipping {config['event']}: {source_dir} not found")
            continue
        
        print(f"\n📦 Processing: {config['event']}")
        
        # Find all images
        images = list(source_dir.rglob('*.jpg')) + list(source_dir.rglob('*.jpeg')) + list(source_dir.rglob('*.png'))
        
        if not images:
            print(f"   ⚠️  No images found")
            continue
        
        print(f"   Found {len(images)} images")
        
        # Split train/val (85/15)
        split_idx = int(len(images) * 0.85)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to YOLO structure
        for split_name, image_list in [('train', train_images), ('val', val_images)]:
            for img_path in tqdm(image_list, desc=f"   {split_name}"):
                # Copy image
                dest_img = yolo_dir / 'images' / split_name / f"{config['event']}_{img_path.name}"
                shutil.copy2(img_path, dest_img)
                
                # Create label file (full frame detection)
                label_path = yolo_dir / 'labels' / split_name / f"{config['event']}_{img_path.stem}.txt"
                with open(label_path, 'w') as f:
                    # Format: class_id x_center y_center width height (normalized)
                    f.write(f"{config['class_id']} 0.5 0.5 1.0 1.0\n")
                
                if split_name == 'train':
                    total_train += 1
                else:
                    total_val += 1
        
        print(f"   ✅ Train: {len(train_images)}, Val: {len(val_images)}")
    
    print(f"\n📊 TOTAL: Train={total_train}, Val={total_val}")
    
    # Create data.yaml
    data_yaml = yolo_dir / 'data.yaml'
    yaml_content = f"""# YOLO v11 Dataset Configuration
# Auto-generated for your project

path: {yolo_dir.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: fire
  1: vehicle_accident
  2: fighting

# Dataset info
nc: 3  # number of classes
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Created: {data_yaml}")
    
    return yolo_dir / 'data.yaml'

def main():
    print("\n" + "="*70)
    print("🎯 DATASET PREPARATION FOR YOLO v11")
    print("="*70)
    
    # Step 1: Extract frames from violence videos
    violence_video_dir = Path('datasets/violence_cctv')
    violence_frames_dir = Path('datasets/violence_frames')
    
    if violence_video_dir.exists():
        print("\n📹 Found violence video dataset")
        if not violence_frames_dir.exists() or len(list(violence_frames_dir.glob('*.jpg'))) == 0:
            print("\n🎯 Using SMART extraction: Only frames where events occur")
            print("   - Motion detection to find event segments")
            print("   - Skips static/non-event portions")
            print("   - Saves disk space and training time")
            extract_frames_from_videos(violence_video_dir, violence_frames_dir, frames_per_segment=10)
        else:
            existing = len(list(violence_frames_dir.glob('*.jpg')))
            print(f"✅ Already extracted: {existing:,} frames")
    else:
        print("⚠️  Violence video dataset not found, skipping")
    
    # Step 2: Organize into YOLO format
    data_yaml_path = organize_yolo_dataset()
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ PREPARATION COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\n📁 YOLO Dataset ready: {data_yaml_path}")
    
    print("\n📋 NEXT STEPS:")
    print("\n1. Train YOLO v11:")
    print(f"   python scripts/train_yolo_v11.py --data {data_yaml_path} --epochs 50 --batch 16 --img 640")
    
    print("\n2. Training time estimate:")
    print("   - CPU: 3-5 hours (50 epochs)")
    print("   - GPU: 30-60 minutes (50 epochs)")
    print("   - Google Colab FREE GPU: 45-90 minutes")
    
    print("\n3. After training:")
    print("   - Best model: runs/train/exp/weights/best.pt")
    print("   - Test: python test_cctv_system.py --model runs/train/exp/weights/best.pt")
    
    print("\n💡 TIP: Start with fewer epochs (--epochs 10) to test the pipeline first!")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
