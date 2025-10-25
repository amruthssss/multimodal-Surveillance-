"""
Convert common dataset folders (COCO-like or VOC-like) into YOLO v11 training format.
Usage: python scripts/convert_to_yolo.py --src path/to/dataset --dst data/yolo_dataset --labels events.txt
Creates: data/yolo_dataset/images/(train|val), data/yolo_dataset/labels/(train|val), data/yolo_dataset/data.yaml
"""

import argparse
from pathlib import Path
import shutil
import random
import os

SPLIT_RATIO = 0.85

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Source dataset root (images/labels per class or COCO-like)')
    p.add_argument('--dst', required=True, help='Destination YOLO dataset folder')
    p.add_argument('--labels', required=True, help='File containing event labels, one per line')
    return p.parse_args()


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def copy_and_split(images, dst_images_train, dst_images_val, labels_map):
    random.shuffle(images)
    split = int(len(images) * SPLIT_RATIO)
    train = images[:split]
    val = images[split:]

    for src, dst in [(train, dst_images_train), (val, dst_images_val)]:
        ensure_dir(dst)
        for img_path in src:
            shutil.copy(img_path, Path(dst) / Path(img_path).name)


if __name__ == '__main__':
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    labels_file = Path(args.labels)

    if not src.exists():
        print(f"Source not found: {src}")
        exit(1)
    if not labels_file.exists():
        print(f"Labels file not found: {labels_file}")
        exit(1)

    labels = [l.strip() for l in open(labels_file, 'r', encoding='utf-8').read().splitlines() if l.strip()]
    classes = {l: i for i, l in enumerate(labels)}

    # Create folders
    images_train = dst / 'images' / 'train'
    images_val = dst / 'images' / 'val'
    labels_train = dst / 'labels' / 'train'
    labels_val = dst / 'labels' / 'val'
    ensure_dir(images_train)
    ensure_dir(images_val)
    ensure_dir(labels_train)
    ensure_dir(labels_val)

    # Assumption: src contains subfolders named after classes with images inside
    images_all = []
    for cls in labels:
        cls_folder = src / cls
        if not cls_folder.exists():
            print(f"Warning: class folder missing: {cls_folder}")
            continue
        for img_file in cls_folder.glob('**/*'):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            # Create a label file for each image in YOLO format (class x_center y_center w h) normalized
            images_all.append(str(img_file))
            # For conversion we expect existing bbox labels elsewhere; here we only copy images and let user create labels

    if len(images_all) == 0:
        print("No images found in source class folders. Consider providing COCO or VOC and implementing parser.")
        exit(1)

    # Copy & split
    copy_and_split(images_all, images_train, images_val, classes)

    # Create a basic data.yaml
    data_yaml = dst / 'data.yaml'
    with open(data_yaml, 'w', encoding='utf-8') as f:
        f.write(f"path: {dst}\n")
        f.write(f"train: {images_train}\n")
        f.write(f"val: {images_val}\n")
        f.write(f"names: {labels}\n")

    print(f"Created YOLO dataset at: {dst}")
    print("Next: prepare labels in `labels/train` and `labels/val` in YOLO format (one .txt per image)")
