#!/usr/bin/env python3
"""
train_yolov11.py

Lightweight wrapper around Ultralytics YOLO training API.
Optimized for 4-class CCTV event detection (fire, vehicle_accident, fighting, explosion)

Features:
- CLI for common training params (data, model, epochs, batch, imgsz, project/name)
- Optional automatic data.yaml creation and simple train/val split when given a dataset folder
- Optional ONNX export after training
- Optimized hyperparameters for 95%+ mAP on CCTV datasets

Recommended training command for RTX 4060 (8GB VRAM):
  python train_yolov11.py --data yolo_ready_final --model yolov11m.pt --epochs 150 --batch 8 --accumulate 4 --imgsz 640 --auto-create

For more VRAM (16GB+):
  python train_yolov11.py --data yolo_ready_final --model yolov11n.pt --epochs 200 --batch 16 --accumulate 2 --imgsz 640 --auto-create

Usage examples:
  # Local RTX 4060 (PowerShell)
  python .\train_yolov11.py --data .\yolo_ready_final --model yolov11m.pt --epochs 150 --batch 8 --accumulate 4 --auto-create

  # Colab (after extraction)
  python /content/train_yolov11.py --data /content/yolo_ready_final --model yolov11m.pt --epochs 150 --batch 16

Requires: pip install ultralytics
"""

import argparse
import random
import shutil
from pathlib import Path
import sys


def collect_label_classes(label_paths):
    classes = set()
    for p in label_paths:
        try:
            with open(p, 'r') as f:
                for line in f:
                    toks = line.strip().split()
                    if not toks:
                        continue
                    try:
                        classes.add(int(toks[0]))
                    except Exception:
                        pass
        except Exception:
            pass
    return sorted(classes)


def write_data_yaml(train_path: Path, val_path: Path, classes_sorted, out_path: Path):
    nc = len(classes_sorted)
    names = [f'class{c}' for c in range(nc)]
    content_lines = [
        f"train: {str(train_path)}",
        f"val:   {str(val_path)}",
        f"nc: {nc}",
        "names: " + str(names)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(content_lines))
    print(f"✅ Wrote data.yaml -> {out_path}")
    return out_path


def auto_create_data_yaml(dataset_root: Path, out_yaml: Path, val_ratio: float = 0.2):
    """Try to create a YOLO data.yaml in out_yaml from dataset_root layout.

    Supports:
    - images/train + images/val + labels/train + labels/val (just writes yaml)
    - flat images/ + labels/ (auto-splits into images/{train,val}, labels/{train,val})
    - images/train + labels/train without val (creates val by copying subset)
    """
    imgs_train = dataset_root / 'images' / 'train'
    imgs_val = dataset_root / 'images' / 'val'
    lbls_train = dataset_root / 'labels' / 'train'
    lbls_val = dataset_root / 'labels' / 'val'
    flat_imgs = dataset_root / 'images'
    flat_lbls = dataset_root / 'labels'

    if imgs_train.exists() and imgs_val.exists() and lbls_train.exists() and lbls_val.exists():
        print('Found standard YOLO layout; creating data.yaml pointing to train/val')
        label_files = list(lbls_train.glob('*.txt')) + list(lbls_val.glob('*.txt'))
        classes = collect_label_classes(label_files)
        return write_data_yaml(imgs_train.resolve(), imgs_val.resolve(), classes, out_yaml)

    # flat case
    if flat_imgs.exists() and flat_lbls.exists():
        print('Found flat images/ and labels/ — creating train/val split (copying files)...')
        all_images = sorted([p for p in flat_imgs.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and p.is_file()])
        pairs = []
        for img in all_images:
            lbl = flat_lbls / (img.stem + '.txt')
            if lbl.exists():
                pairs.append((img, lbl))
        if not pairs:
            raise SystemExit('No labels found matching images — cannot create data.yaml')
        random.seed(42)
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

        t_imgs = dataset_root / 'images' / 'train'
        v_imgs = dataset_root / 'images' / 'val'
        t_lbls = dataset_root / 'labels' / 'train'
        v_lbls = dataset_root / 'labels' / 'val'
        for d in (t_imgs, v_imgs, t_lbls, v_lbls):
            d.mkdir(parents=True, exist_ok=True)

        for img, lbl in train_pairs:
            shutil.copy2(img, t_imgs / img.name)
            shutil.copy2(lbl, t_lbls / lbl.name)
        for img, lbl in val_pairs:
            shutil.copy2(img, v_imgs / img.name)
            shutil.copy2(lbl, v_lbls / lbl.name)

        label_files = list(t_lbls.glob('*.txt')) + list(v_lbls.glob('*.txt'))
        classes = collect_label_classes(label_files)
        return write_data_yaml(t_imgs.resolve(), v_imgs.resolve(), classes, out_yaml)

    # images/train & labels/train but no val
    if imgs_train.exists() and lbls_train.exists() and not imgs_val.exists():
        print('Found images/train + labels/train, creating val split by copying subset...')
        train_images = [p for p in imgs_train.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'} and p.is_file()]
        pairs = []
        for img in train_images:
            lbl = lbls_train / (img.stem + '.txt')
            if lbl.exists():
                pairs.append((img, lbl))
        if not pairs:
            raise SystemExit('No label files found in labels/train — cannot create val split')
        random.seed(42)
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_ratio))
        val_pairs = pairs[:n_val]
        imgs_val.mkdir(parents=True, exist_ok=True)
        lbls_val.mkdir(parents=True, exist_ok=True)
        for img, lbl in val_pairs:
            shutil.copy2(img, imgs_val / img.name)
            shutil.copy2(lbl, lbls_val / lbl.name)
        label_files = list(lbls_train.glob('*.txt')) + list(lbls_val.glob('*.txt'))
        classes = collect_label_classes(label_files)
        return write_data_yaml(imgs_train.resolve(), imgs_val.resolve(), classes, out_yaml)

    raise SystemExit('Unsupported dataset layout — please provide a folder with images/ and labels/ or a proper YOLO structure')


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLOv11 (Ultralytics) with helpful defaults for CCTV datasets')
    p.add_argument('--data', required=True, help='path to data.yaml or dataset root (folder containing images/ and labels/)')
    p.add_argument('--model', default='yolov11m.pt', help='backbone weights (yolov11m.pt or yolov11l.pt)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--project', default='runs/detect')
    p.add_argument('--name', default='yolo_v11_95_percent')
    p.add_argument('--device', default='0', help='device id or cpu')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--cache', action='store_true', help='cache images in RAM')
    p.add_argument('--auto-create', action='store_true', help='auto-create data.yaml when given a dataset folder with images/labels')
    p.add_argument('--accumulate', type=int, default=1, help='gradient accumulation steps (simulate larger batch size)')
    p.add_argument('--weights-url', type=str, default=None, help='optional URL to download model weights if local file not found')
    p.add_argument('--export-onnx', action='store_true', help='Export best.pt to ONNX after training')
    p.add_argument('--exist-ok', action='store_true', help='Pass exist_ok to ultralytics train')
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)

    # determine data.yaml
    if data_path.is_file() and data_path.suffix in {'.yaml', '.yml'}:
        yaml_path = data_path
    elif data_path.is_dir():
        yaml_path = data_path / 'data.yaml'
        if not yaml_path.exists():
            if args.auto_create:
                print('Auto-creating data.yaml from dataset folder...')
                yaml_path = auto_create_data_yaml(data_path, yaml_path)
            else:
                raise SystemExit(f"data.yaml not found at {yaml_path}. Use --auto-create to attempt to generate it.")
    else:
        raise SystemExit('Provided --data path must be a data.yaml file or a dataset folder')

    # Lazy import ultralytics (so script can be inspected without installing)
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('Error importing ultralytics. Install with: pip install ultralytics')
        raise

    print(f"📥 Loading model: {args.model}")
    
    # Auto-download YOLOv11 weights if not present
    model_path = Path(args.model)
    if not model_path.exists():
        # Try ultralytics hub auto-download first (easiest)
        print(f"⬇️  Model not found locally. Attempting auto-download from Ultralytics...")
        try:
            # YOLO() will auto-download from ultralytics if given standard name
            model = YOLO(args.model)
            print(f"✅ Successfully loaded {args.model}")
        except Exception as e1:
            # Fallback: try manual download if --weights-url provided
            if args.weights_url:
                import urllib.request
                try:
                    print(f"⬇️  Downloading from custom URL: {args.weights_url}")
                    urllib.request.urlretrieve(args.weights_url, str(model_path))
                    print("✅ Download complete")
                    model = YOLO(str(model_path))
                except Exception as e2:
                    print(f"❌ Failed to download from URL: {e2}")
                    raise
            else:
                print(f"❌ Model '{args.model}' not found.")
                print(f"   Ultralytics auto-download failed: {e1}")
                print(f"\n💡 Solutions:")
                print(f"   1. Use standard YOLO names: yolov11n.pt, yolov11s.pt, yolov11m.pt, yolov11l.pt, yolov11x.pt")
                print(f"   2. Download manually: https://github.com/ultralytics/assets/releases/")
                print(f"   3. Provide custom URL: --weights-url https://...")
                raise FileNotFoundError(f"{model_path} does not exist")
    else:
        model = YOLO(str(model_path))

    # Optimized hyperparameters for 4-class CCTV event detection (95%+ mAP target)
    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        patience=30,  # Increased patience for better convergence
        imgsz=args.imgsz,
        batch=args.batch,
        accumulate=args.accumulate,
        optimizer='AdamW',  # AdamW for better generalization
        lr0=0.002,  # Lower initial LR for stable convergence
        lrf=0.001,  # Lower final LR for fine-tuning
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,  # Longer warmup for stability
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        # Loss weights optimized for multi-class CCTV
        box=8.0,  # Higher box weight for precise localization
        cls=1.0,  # Increased cls weight for 4-class distinction
        dfl=1.5,
        # Augmentation tuned for CCTV event frames
        hsv_h=0.01,  # Reduced hue shift (CCTV color consistency)
        hsv_s=0.5,  # Moderate saturation
        hsv_v=0.3,  # Reduced brightness (varied lighting)
        degrees=5.0,  # Slight rotation (camera angles)
        translate=0.1,
        scale=0.5,
        shear=0.0,  # No shear (rectangular CCTV frames)
        perspective=0.0001,  # Minimal perspective (fixed cameras)
        flipud=0.0,  # No vertical flip (gravity matters for events)
        fliplr=0.5,  # Horizontal flip OK
        mosaic=1.0,  # Full mosaic for multi-scale learning
        mixup=0.1,  # Light mixup for regularization
        copy_paste=0.1,  # Light copy-paste for object variety
        # Training settings
        device=args.device,
        workers=args.workers,
        cache='disk',  # Cache to disk for 3K+ images
        amp=True,  # Mixed precision for RTX 4060
        close_mosaic=15,  # Close mosaic later for better accuracy
        val=True,
        save=True,
        save_period=20,  # Save every 20 epochs
        plots=True,
        project=str(args.project),
        name=str(args.name),
        exist_ok=bool(args.exist_ok),
        verbose=True,
        # Additional optimizations
        seed=42,  # Reproducibility
        deterministic=False,  # Allow cudnn autotuner for speed
        single_cls=False,  # Multi-class detection
        rect=True,  # Rectangular training for faster convergence
        resume=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val_period=1,  # Validate every epoch
    )

    print('\n🚀 Starting training with the following configuration:')
    for k, v in train_kwargs.items():
        print(f'  {k}: {v}')

    results = model.train(**train_kwargs)
    print('\n✅ TRAINING COMPLETE!')

    best = Path(train_kwargs['project']) / train_kwargs['name'] / 'weights' / 'best.pt'
    if args.export_onnx and best.exists():
        print('📦 Exporting to ONNX...')
        model_export = YOLO(str(best))
        model_export.export(format='onnx')
        print('✅ Exported to ONNX')


if __name__ == '__main__':
    main()
