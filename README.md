# Multi-Modal Surveillance Detection System 🚨

AI-powered real-time surveillance system that detects accidents, explosions, fire, smoke, and fighting using hybrid detection (YOLOv11 + AI Agent).

> **Note**: This is the public version. Pre-trained models and pattern files are not included due to size constraints. See training instructions below.

## 🎯 Features

- **🚗 Vehicle Accident Detection**: Detects collisions with motion spike analysis (200-450 range)
- **💥 Explosion Detection**: Identifies sudden bursts with radial motion and bright flashes
- **🔥 Fire Detection**: Recognizes flames with color analysis and flickering patterns
- **💨 Smoke Detection**: Detects gray/black plumes with slow drift patterns
- **👊 Fighting Detection**: Identifies physical altercations with irregular human motion

## 🧠 Technology Stack

- **YOLOv11m**: Custom trained model (75.5% mAP@50, 4 classes)
- **AI Agent**: Pattern-based detection with 221,660+ learned patterns
- **Hybrid Fusion**: 80% AI Agent + 20% YOLO weighted detection
- **Optical Flow**: Farneback method for motion analysis
- **Morphological Filtering**: HSV-based color segmentation
- **Temporal Validation**: Streak requirement to prevent false positives

## 📋 Requirements

```txt
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
ultralytics>=8.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
requests>=2.31.0
beautifulsoup4>=4.12.0
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/amruthssss/ss.git
cd ss

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Model

Place your trained YOLOv11 model in:
```
runs/detect/train/weights/best.pt
```

Or train your own model with the provided dataset structure.

### 3. Run Detection

```bash
# Interactive mode
python main.py

# Or specify video directly
python main.py --video path/to/video.mp4
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | 75.5% |
| Precision | 75.6% |
| Recall | 72.8% |
| Classes | 4 (fire, accident, fighting, explosion) |
| Epochs | 130 |

## 🎨 Detection Output

The system displays:
- **White text alerts**: "ALERT: [EVENT]!"
- **Confidence score**: "Confidence: XX.X%"
- **Frame number**: "Frame: X/Y"
- **Clean video**: No colored overlays, just text

## 🔧 Configuration

### Detection Thresholds

Edit `enhanced_final_ultra_system.py`:

```python
self.thresholds = {
    'accident': {
        'motion_spike': 200,        # Motion spike range: 200-450
        'confidence': 65.0          # Detection confidence
    },
    'explosion': {
        'brightness_spike': 180,
        'confidence': 70.0
    },
    # ... other events
}
```

### Streak Validation

```python
self.streak_requirement = 5  # Frames needed to confirm alert
```

## 📂 Project Structure

```
├── main.py                          # Main entry point
├── enhanced_final_ultra_system.py   # Core detection system
├── smart_event_differentiator.py    # Event differentiation logic
├── web_pattern_learner.py          # Pattern learning from web
├── learn_from_patterns.py          # Threshold optimization
├── models/                          # Trained models & patterns
│   ├── best.pt                     # YOLOv11 weights
│   └── *.pkl                       # Learned patterns (221K+)
├── test_*.py                        # Test scripts
└── requirements.txt                 # Dependencies
```

## 🧪 Testing

```bash
# Test on sample video
python test_dtest.py

# Test accident detection
python test_road_accident.py

# Learn from collected patterns
python learn_from_patterns.py
```

## 🎓 How It Works

### 1. Hybrid Detection
- **YOLO**: Fast object detection for vehicles, fire, smoke
- **AI Agent**: Pattern-based analysis with learned characteristics
- **Fusion**: Weighted combination (80% Agent + 20% YOLO)

### 2. Event Differentiation

```python
# Accident vs Explosion
if motion_spike > 450: → Explosion
elif 200 < motion_spike < 450: → Accident
else: → Normal traffic

# Fire vs Explosion
if duration < 1s + bright flash: → Explosion
elif duration > 3s + flickering: → Fire
```

### 3. Temporal Validation
- Requires 5 consecutive frames above threshold
- Prevents single-frame false positives
- Validates event duration patterns

## 📈 Performance

- **Processing Speed**: ~2-3 FPS (with frame skipping)
- **Frame Skip**: Every 2nd frame processed
- **Accuracy**: 221,660+ learned patterns from web sources
- **False Positives**: Minimized with strict thresholds

## 🔍 Event Detection Criteria

### Accident 🚗
- Motion spike: 200-450
- Vehicles detected (YOLO confidence > 20%)
- NO explosion colors (orange/yellow)
- Linear collision pattern

### Explosion 💥
- Motion spike: 450+
- Bright flash (brightness > 180)
- Orange/yellow colors (> 5%)
- Radial burst pattern

### Fire 🔥
- Orange flames (> 12%)
- Sustained flickering (> 3s)
- Upward motion
- High brightness

### Smoke 💨
- Gray coverage (> 28%)
- Slow drift pattern
- Low brightness
- Gradual increase

### Fighting 👊
- Multiple humans detected
- Irregular motion (60-200)
- Sustained activity (> 18 frames)
- No vehicles dominant

## 🛠️ Training Custom Model

```bash
# Prepare dataset in YOLO format
# Train YOLOv11
yolo train model=yolov11m.pt data=data.yaml epochs=130 imgsz=640
```

## 📝 Output Files

- **Video**: `[filename]_detected.mp4`
- **Feature Log**: `enhanced_features_log.csv`
- **Learned Thresholds**: `models/learned_optimal_thresholds.json`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is for educational and research purposes.

## 👤 Author

**Amruth**
- GitHub: [@amruthssss](https://github.com/amruthssss)

## 🙏 Acknowledgments

- YOLOv11 by Ultralytics
- Pattern learning from Google Images, Bing, Yahoo
- OpenCV for computer vision
- PyTorch for deep learning

## 📞 Support

For issues or questions, please open an issue on GitHub.

---

**⚠️ Note**: This system is designed for surveillance and safety monitoring. Ensure compliance with local privacy laws and regulations when deploying.
