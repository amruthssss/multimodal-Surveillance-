"""
🎯 Test Your Trained YOLOv11 Model
===================================
Tests the trained model on images/videos and shows results.
"""

from ultralytics import YOLO
import cv2
from pathlib import Path

# Class names for your 4-class CCTV model
CLASS_NAMES = {
    0: 'fire',
    1: 'vehicle_accident', 
    2: 'fighting',
    3: 'explosion'
}

def test_trained_model():
    """Test the trained YOLOv11 model."""
    
    print("="*60)
    print("🎯 TESTING TRAINED YOLOv11 MODEL")
    print("="*60)
    
    # Load your trained model
    model_path = "runs/detect/train/weights/best.pt"
    print(f"\n📦 Loading model: {model_path}")
    
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        
        # Print model info
        print(f"\n📊 Model Info:")
        print(f"   Classes: {len(CLASS_NAMES)}")
        for idx, name in CLASS_NAMES.items():
            print(f"   {idx}: {name}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test options
    print("\n" + "="*60)
    print("🎥 SELECT TEST MODE:")
    print("="*60)
    print("1. Test on validation images (from yolo_ready_final/images/val)")
    print("2. Test on webcam (real-time)")
    print("3. Test on video file")
    print("4. Test on single image")
    print("5. View training results")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        # Test on validation set
        val_dir = Path("yolo_ready_final/images/val")
        if not val_dir.exists():
            print(f"❌ Validation directory not found: {val_dir}")
            return
        
        print(f"\n🔍 Testing on validation images...")
        results = model.val(data="yolo_ready_final/data.yaml", split="val")
        
        print("\n📊 VALIDATION RESULTS:")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        
        # Test on a few sample images
        images = list(val_dir.glob("*.jpg"))[:5]
        print(f"\n🖼️ Testing on {len(images)} sample images...")
        
        for img_path in images:
            result = model.predict(str(img_path), conf=0.25, save=True)
            print(f"   ✓ {img_path.name}: {len(result[0].boxes)} detections")
        
        print(f"\n💾 Results saved to: runs/detect/predict/")
    
    elif choice == "2":
        # Test on webcam
        print("\n🎥 Starting webcam test...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model.predict(frame, conf=0.25, verbose=False)
            
            # Draw results
            annotated = results[0].plot()
            
            cv2.imshow("YOLOv11 - Trained Model", annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam test complete!")
    
    elif choice == "3":
        # Test on video
        video_path = input("\nEnter video path: ").strip().strip('"').strip("'")
        
        if not Path(video_path).exists():
            print(f"❌ Video not found: {video_path}")
            return
        
        print(f"\n🎬 Processing video: {video_path}")
        results = model.predict(video_path, conf=0.25, save=True)
        print(f"✅ Results saved to: runs/detect/predict/")
    
    elif choice == "4":
        # Test on single image
        img_path = input("\nEnter image path: ").strip().strip('"').strip("'")
        
        if not Path(img_path).exists():
            print(f"❌ Image not found: {img_path}")
            return
        
        print(f"\n🖼️ Processing image: {img_path}")
        results = model.predict(img_path, conf=0.25, save=True)
        
        # Print detections
        boxes = results[0].boxes
        print(f"\n🎯 Detections: {len(boxes)}")
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = CLASS_NAMES.get(cls, f"class_{cls}")
            print(f"   {class_name}: {conf:.2%}")
        
        print(f"\n💾 Result saved to: runs/detect/predict/")
    
    elif choice == "5":
        # View training results
        print("\n📊 TRAINING RESULTS:")
        print("="*60)
        
        # Read results.csv
        results_file = Path("runs/detect/train/weights/results.csv")
        if results_file.exists():
            import pandas as pd
            df = pd.read_csv(results_file)
            
            # Get last row (final epoch)
            last = df.iloc[-1]
            
            print(f"\n🏆 FINAL RESULTS (Epoch {int(last['epoch'])}):")
            print(f"   Box Loss: {last['train/box_loss']:.4f}")
            print(f"   Class Loss: {last['train/cls_loss']:.4f}")
            print(f"   DFL Loss: {last['train/dfl_loss']:.4f}")
            print(f"\n   Precision: {last['metrics/precision(B)']:.3f}")
            print(f"   Recall: {last['metrics/recall(B)']:.3f}")
            print(f"   mAP50: {last['metrics/mAP50(B)']:.3f}")
            print(f"   mAP50-95: {last['metrics/mAP50-95(B)']:.3f}")
            
            # Check if achieved target
            if last['metrics/mAP50(B)'] >= 0.95:
                print("\n🎉 TARGET ACHIEVED! mAP50 >= 95%")
            else:
                print(f"\n⚠️ mAP50: {last['metrics/mAP50(B)']:.1%} (target: 95%)")
        
        print(f"\n📈 Training graphs saved in: runs/detect/train/weights/")
        print("   - results.png (loss and metrics)")
        print("   - confusion_matrix.png")
        print("   - PR_curve.png (Precision-Recall)")
        print("   - F1_curve.png")
    
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    test_trained_model()
