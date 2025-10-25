"""YOLO Wrapper
Handles lazy loading of a YOLOv8 model (ultralytics) for object detection.
Falls back gracefully if weights are missing; can auto-download.
Supports custom-trained models with class mapping.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import os
import sys

try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

# Import class mapping for custom model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
try:
    from class_mapping import (
        CLASS_MAPPING, 
        DISPLAY_NAMES, 
        RISK_LEVELS, 
        CONFIDENCE_THRESHOLDS,
        get_event_name,
        get_display_name
    )
    HAS_CLASS_MAPPING = True
except ImportError:
    HAS_CLASS_MAPPING = False

# Try custom model first, fallback to default
CUSTOM_MODEL = 'models/best.pt'
DEFAULT_MODEL = 'yolov8n.pt'

class YOLOWrapper:
    def __init__(self, weights: str | None = None, auto_download: bool = True):
        # Prefer custom trained model if available
        if weights is None:
            if os.path.exists(CUSTOM_MODEL):
                self.weights = CUSTOM_MODEL
                self.is_custom = True
                print(f"✅ Using custom-trained YOLO model: {CUSTOM_MODEL}")
            else:
                self.weights = DEFAULT_MODEL
                self.is_custom = False
                print(f"⚠️ Custom model not found, using default: {DEFAULT_MODEL}")
        else:
            self.weights = weights
            self.is_custom = 'best.pt' in weights or 'custom' in weights.lower()
        
        self.auto_download = auto_download
        self._model = None
        self._ensure_model()

    def _ensure_model(self):
        if YOLO is None:
            return
        if self._model is None:
            try:
                self._model = YOLO(self.weights)
                print(f"✅ YOLO model loaded: {self.weights}")
            except Exception as e:
                if not self.auto_download:
                    self._model = None
                else:
                    # Attempt download of default
                    try:
                        self._model = YOLO(DEFAULT_MODEL)
                        print(f"⚠️ Loaded default model: {DEFAULT_MODEL}")
                    except Exception:
                        self._model = None
                        print(f"❌ Failed to load YOLO model: {e}")

    def detect(self, frame) -> Dict[str, Any]:
        if self._model is None:
            return {'objects': []}
        try:
            results = self._model.predict(source=frame, verbose=False, imgsz=640)[0]
            objs: List[Dict[str, Any]] = []
            
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                # Add event mapping if custom model
                detection = {
                    'class_id': cls_id,
                    'confidence': conf,
                    'bbox': xyxy
                }
                
                # If using custom model with class mapping
                if self.is_custom and HAS_CLASS_MAPPING:
                    event_name = get_event_name(cls_id)
                    display_name = get_display_name(cls_id)
                    threshold = CONFIDENCE_THRESHOLDS.get(event_name, 0.3)
                    
                    # Only include if above threshold
                    if conf >= threshold:
                        detection['event'] = event_name
                        detection['display_name'] = display_name
                        detection['risk_level'] = RISK_LEVELS.get(event_name, 'MEDIUM')
                        objs.append(detection)
                else:
                    # Default YOLO classes (80 COCO classes)
                    objs.append(detection)
            
            return {'objects': objs}
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return {'objects': []}
