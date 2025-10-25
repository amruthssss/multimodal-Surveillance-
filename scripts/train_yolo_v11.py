"""
Training launcher for YOLO v11 (assumes ultralytics vX with YOLO.train API)
This script prepares a training run given a data.yaml (YOLO format) and a set of hyperparameters.

Usage:
    python scripts/train_yolo_v11.py --data data/yolo_dataset/data.yaml --epochs 50 --batch 16 --img 640

Outputs:
    - Trained weights under `runs/train/exp...`
    - Logs and best.pt

Note: This is a helper wrapper; ensure you have `ultralytics` installed and GPU available for faster training.
"""

import argparse
from ultralytics import YOLO
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to data.yaml')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--img', type=int, default=640)
    p.add_argument('--project', default='runs/train')
    p.add_argument('--name', default='yolov11_exp')
    p.add_argument('--device', default='0')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_yaml = args.data

    if not Path(data_yaml).exists():
        print(f"data.yaml not found: {data_yaml}")
        exit(1)

    print(f"Starting YOLOv11 training: data={data_yaml} epochs={args.epochs} batch={args.batch} img={args.img}")

    # Create a new model or load base
    model = YOLO('yolov11n.pt') if Path('yolov11n.pt').exists() else YOLO('yolov8n.pt')

    # Train
    model.train(data=data_yaml,
                epochs=args.epochs,
                imgsz=args.img,
                batch=args.batch,
                project=args.project,
                name=args.name,
                device=args.device)

    print('Training complete. Check runs/train for results.')
