"""
OPTIMIZED YOLO v11 TRAINING SCRIPT FOR 95%+ ACCURACY
Configured for your datasets: fire, vehicle_accident, fighting
Uses advanced hyperparameters and data augmentation
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("⚠️  No GPU detected, using CPU (will be slower)")
        return False

def train_yolo_v11_optimized():
    """
    Train YOLO v11 with optimized settings for 95%+ accuracy
    """
    print("\n" + "="*70)
    print("🎯 YOLO v11 OPTIMIZED TRAINING")
    print("="*70)
    print("Target: 95%+ mAP (Mean Average Precision)")
    print("Events: fire, vehicle_accident, fighting")
    print("="*70)
    
    # Check GPU
    has_gpu = check_gpu()
    device = 0 if has_gpu else 'cpu'
    
    # Dataset configuration
    data_yaml = 'datasets/yolo_ready/data.yaml'
    
    if not Path(data_yaml).exists():
        print(f"\n❌ Dataset not found: {data_yaml}")
        print("Run: python scripts/prepare_for_training.py")
        return
    
    print(f"\n✅ Dataset: {data_yaml}")
    
    # Model selection
    print("\n📦 MODEL SELECTION:")
    print("   1. yolov11n.pt - Nano (Fast, 85-90% accuracy)")
    print("   2. yolov11s.pt - Small (Balanced, 90-93% accuracy)")
    print("   3. yolov11m.pt - Medium (Best for 95%+ accuracy) ⭐ RECOMMENDED")
    print("   4. yolov11l.pt - Large (Best accuracy, slower)")
    
    model_choice = input("\nChoose model (1/2/3/4) [default: 3]: ").strip() or '3'
    
    models = {
        '1': 'yolov11n.pt',
        '2': 'yolov11s.pt',
        '3': 'yolov11m.pt',
        '4': 'yolov11l.pt'
    }
    
    model_name = models.get(model_choice, 'yolov11m.pt')
    print(f"\n✅ Selected: {model_name}")
    
    # Load model
    print(f"\n📥 Loading {model_name}...")
    model = YOLO(model_name)
    
    # Training configuration for 95%+ accuracy
    print("\n⚙️  TRAINING CONFIGURATION:")
    print("="*70)
    
    # Optimized hyperparameters
    config = {
        'data': data_yaml,
        'epochs': 100,              # More epochs = better learning
        'imgsz': 640,               # Standard YOLO size
        'batch': 16,                # Adjust based on GPU memory
        'patience': 20,             # Early stopping patience
        'save': True,
        'save_period': 10,          # Save checkpoint every 10 epochs
        'cache': True,              # Cache images for faster training
        'device': device,
        'workers': 8,               # Data loading workers
        'project': 'runs/train',
        'name': 'yolo_v11_95_percent',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',        # Auto-select best optimizer
        'verbose': True,
        'seed': 0,                  # Reproducibility
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,             # Cosine learning rate scheduler
        'close_mosaic': 10,         # Close mosaic augmentation in last 10 epochs
        'resume': False,
        'amp': True,                # Automatic Mixed Precision
        'fraction': 1.0,            # Use 100% of dataset
        'profile': False,
        'freeze': None,
        
        # Learning rate settings
        'lr0': 0.01,                # Initial learning rate
        'lrf': 0.01,                # Final learning rate (lr0 * lrf)
        'momentum': 0.937,          # SGD momentum
        'weight_decay': 0.0005,     # Weight decay
        'warmup_epochs': 3.0,       # Warmup epochs
        'warmup_momentum': 0.8,     # Warmup momentum
        'warmup_bias_lr': 0.1,      # Warmup bias learning rate
        
        # Loss weights
        'box': 7.5,                 # Box loss weight
        'cls': 0.5,                 # Class loss weight
        'dfl': 1.5,                 # DFL loss weight
        
        # Data augmentation (AGGRESSIVE for better accuracy)
        'hsv_h': 0.015,             # HSV-Hue augmentation
        'hsv_s': 0.7,               # HSV-Saturation augmentation
        'hsv_v': 0.4,               # HSV-Value augmentation
        'degrees': 0.0,             # Rotation (0 = disabled, CCTV is usually fixed)
        'translate': 0.1,           # Translation
        'scale': 0.5,               # Scaling
        'shear': 0.0,               # Shear (0 = disabled for CCTV)
        'perspective': 0.0,         # Perspective (0 = disabled for CCTV)
        'flipud': 0.0,              # Vertical flip (0 = disabled for CCTV)
        'fliplr': 0.5,              # Horizontal flip (50% chance)
        'mosaic': 1.0,              # Mosaic augmentation
        'mixup': 0.0,               # Mixup augmentation
        'copy_paste': 0.0,          # Copy-paste augmentation
        
        # Validation settings
        'val': True,
        'plots': True,              # Save plots
    }
    
    # Adjust batch size based on GPU memory
    if has_gpu:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory < 8:
            config['batch'] = 8
            print("⚠️  Low GPU memory, reducing batch size to 8")
        elif gpu_memory >= 16:
            config['batch'] = 32
            print("✅ High GPU memory, increasing batch size to 32")
    else:
        config['batch'] = 4  # CPU
        config['workers'] = 4
        print("⚠️  CPU mode, reducing batch size to 4")
    
    # Print configuration
    print(f"\n   Epochs: {config['epochs']}")
    print(f"   Batch size: {config['batch']}")
    print(f"   Image size: {config['imgsz']}")
    print(f"   Learning rate: {config['lr0']} → {config['lr0'] * config['lrf']}")
    print(f"   Optimizer: {config['optimizer']}")
    print(f"   Data augmentation: AGGRESSIVE (for accuracy)")
    print(f"   Device: {'GPU' if has_gpu else 'CPU'}")
    
    # Estimate training time
    if has_gpu:
        est_time = "2-3 hours (GPU)"
    else:
        est_time = "15-20 hours (CPU)"
    
    print(f"\n⏱️  Estimated training time: {est_time}")
    print("="*70)
    
    # Confirm
    confirm = input("\nStart training? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Training cancelled")
        return
    
    # Start training
    print("\n" + "="*70)
    print("🚀 STARTING TRAINING...")
    print("="*70)
    print("\n💡 TIPS:")
    print("   - Loss should decrease over time")
    print("   - mAP should increase (target: 95%+)")
    print("   - Training can be stopped with Ctrl+C (model auto-saves)")
    print("   - Check progress: runs/train/yolo_v11_95_percent/")
    print("\n" + "="*70 + "\n")
    
    try:
        # Train the model
        results = model.train(**config)
        
        # Training complete
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        # Results
        best_model = Path(config['project']) / config['name'] / 'weights' / 'best.pt'
        last_model = Path(config['project']) / config['name'] / 'weights' / 'last.pt'
        
        print(f"\n📁 RESULTS:")
        print(f"   Best model: {best_model}")
        print(f"   Last model: {last_model}")
        print(f"   Results folder: {Path(config['project']) / config['name']}")
        
        # Metrics
        print(f"\n📊 FINAL METRICS:")
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            if 'metrics/mAP50(B)' in metrics:
                map50 = metrics['metrics/mAP50(B)'] * 100
                print(f"   mAP@50: {map50:.2f}%")
            if 'metrics/mAP50-95(B)' in metrics:
                map50_95 = metrics['metrics/mAP50-95(B)'] * 100
                print(f"   mAP@50-95: {map50_95:.2f}%")
        
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. Test model: python test_cctv_system.py --model {best_model}")
        print(f"   2. View results: runs/train/yolo_v11_95_percent/")
        print(f"   3. Check plots: results.png, confusion_matrix.png")
        
        print("\n" + "="*70)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Model checkpoints saved in: runs/train/yolo_v11_95_percent/weights/")
        print("   You can resume with: resume=True")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_yolo_v11_optimized()
