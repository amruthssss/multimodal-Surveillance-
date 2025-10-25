"""
Calculate YOLO v11 training time estimates
Based on your downloaded datasets
"""

def estimate_training_time():
    print("="*70)
    print("⏱️  YOLO v11 TRAINING TIME ESTIMATES")
    print("="*70)
    
    # Your datasets
    datasets = {
        'fire': 999,
        'accidents': 989,
        'fighting': 36320  # 3632 videos × 10 frames per event segment
    }
    
    total_images = sum(datasets.values())
    
    print(f"\n📊 YOUR DATASETS:")
    for event, count in datasets.items():
        print(f"   {event}: {count:,} images")
    print(f"   TOTAL: {total_images:,} images")
    
    # Training parameters
    epochs = 50
    batch_size = 16
    img_size = 640
    
    # Calculate iterations
    iterations_per_epoch = total_images // batch_size
    total_iterations = iterations_per_epoch * epochs
    
    print(f"\n⚙️  TRAINING CONFIGURATION:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}x{img_size}")
    print(f"   Iterations per epoch: {iterations_per_epoch:,}")
    print(f"   Total iterations: {total_iterations:,}")
    
    # Time estimates (based on typical YOLO training speeds)
    print(f"\n{'='*70}")
    print("⏱️  TIME ESTIMATES:")
    print(f"{'='*70}")
    
    # CPU times (ms per iteration)
    cpu_ms_per_iter = 800  # ~0.8 seconds per iteration on modern CPU
    cpu_total_minutes = (total_iterations * cpu_ms_per_iter) / 1000 / 60
    
    print(f"\n🖥️  CPU Training (Intel i5/i7 or AMD Ryzen 5/7):")
    print(f"   Time per iteration: ~{cpu_ms_per_iter}ms")
    print(f"   Total time: {cpu_total_minutes:.0f} minutes ({cpu_total_minutes/60:.1f} hours)")
    print(f"   Estimated: 🔴 {cpu_total_minutes/60:.1f} - {cpu_total_minutes/60*1.2:.1f} hours")
    
    # GPU times
    gpu_configs = [
        ("RTX 3060/3070", 50),
        ("RTX 3080/3090", 35),
        ("RTX 4070/4080", 30),
        ("RTX 4090", 20),
        ("Google Colab Free (T4)", 60),
        ("Google Colab Pro (V100)", 40)
    ]
    
    print(f"\n🎮 GPU Training:")
    for gpu_name, ms_per_iter in gpu_configs:
        gpu_minutes = (total_iterations * ms_per_iter) / 1000 / 60
        print(f"   {gpu_name:25s}: {gpu_minutes:.0f} min ({gpu_minutes/60:.1f} hours)")
    
    # Recommended approach
    print(f"\n{'='*70}")
    print("💡 RECOMMENDATIONS:")
    print(f"{'='*70}")
    
    print(f"\n✅ BEST OPTION: Google Colab FREE GPU")
    print(f"   - Cost: FREE")
    print(f"   - Time: ~1.5 hours (50 epochs)")
    print(f"   - GPU: Tesla T4 (16GB)")
    print(f"   - Steps:")
    print(f"      1. Upload datasets to Google Drive (~2 GB)")
    print(f"      2. Open Colab: https://colab.research.google.com/")
    print(f"      3. Enable GPU: Runtime → Change runtime type → GPU")
    print(f"      4. Run training commands")
    
    print(f"\n⚠️  IF USING YOUR PC:")
    print(f"   CPU: {cpu_total_minutes/60:.1f} hours (overnight training)")
    print(f"   GPU: Check your GPU model above")
    
    print(f"\n🚀 QUICK TEST (10 epochs instead of 50):")
    quick_time_cpu = cpu_total_minutes / 5
    quick_time_gpu = (total_iterations * 60) / 1000 / 60 / 5  # Colab
    print(f"   CPU: {quick_time_cpu:.0f} minutes ({quick_time_cpu/60:.1f} hours)")
    print(f"   Colab GPU: {quick_time_gpu:.0f} minutes")
    print(f"   Good for testing pipeline before full training")
    
    # Training commands
    print(f"\n{'='*70}")
    print("📋 TRAINING COMMANDS:")
    print(f"{'='*70}")
    
    print(f"\n1. Quick test (10 epochs, ~20 min on GPU):")
    print(f"   python scripts/train_yolo_v11.py --data datasets/yolo_ready/data.yaml --epochs 10 --batch 16")
    
    print(f"\n2. Full training (50 epochs):")
    print(f"   python scripts/train_yolo_v11.py --data datasets/yolo_ready/data.yaml --epochs 50 --batch 16")
    
    print(f"\n3. Production training (100 epochs, best accuracy):")
    print(f"   python scripts/train_yolo_v11.py --data datasets/yolo_ready/data.yaml --epochs 100 --batch 16")
    
    print(f"\n{'='*70}")
    print("💾 STORAGE REQUIREMENTS:")
    print(f"{'='*70}")
    print(f"   Datasets: ~2 GB")
    print(f"   Training outputs: ~500 MB")
    print(f"   Total: ~2.5 GB free space needed")
    
    print(f"\n{'='*70}")
    
    # Google Colab instructions
    print("\n📖 GOOGLE COLAB TRAINING GUIDE:")
    print("="*70)
    print("""
1. Prepare datasets locally:
   python scripts/prepare_for_training.py
   
2. Zip the dataset:
   Compress-Archive -Path datasets/yolo_ready -DestinationPath yolo_dataset.zip
   
3. Upload to Google Drive:
   - Go to: https://drive.google.com
   - Upload yolo_dataset.zip
   
4. Open Google Colab:
   - Go to: https://colab.research.google.com/
   - New Notebook
   
5. Enable GPU:
   Runtime → Change runtime type → Hardware accelerator → GPU → Save
   
6. In Colab, run:
   
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Install ultralytics
   !pip install ultralytics
   
   # Unzip dataset
   !unzip /content/drive/MyDrive/yolo_dataset.zip -d /content/
   
   # Train YOLO v11
   !yolo train data=/content/yolo_ready/data.yaml model=yolov11n.pt epochs=50 imgsz=640 batch=16
   
   # Download trained model
   from google.colab import files
   files.download('runs/detect/train/weights/best.pt')

7. Download best.pt and use it!
   
""")
    
    print("="*70)

if __name__ == "__main__":
    estimate_training_time()
