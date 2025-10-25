"""
SMART FUSION ENGINE - Learning Mode
====================================
Uses models intelligently:
- YOLO: Primary detector (every frame, fast)
- Other models: Learning & validation (periodic, boost accuracy)

Strategy:
1. YOLO runs on EVERY frame (fast, 30 FPS)
2. Other models run every N frames (learning)
3. Models boost YOLO confidence when uncertain
4. System learns patterns over time
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import deque
import threading

# Import all model wrappers
from utils.yolo_wrapper import YOLOWrapper
from utils.emotion_wrapper import EmotionWrapper
from utils.action_wrapper import ActionWrapper
from utils.audio_wrapper import AudioWrapper
from utils.svm_wrapper import SVMWrapper

# Import class mapping
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from class_mapping import (
    CLASS_MAPPING, 
    DISPLAY_NAMES, 
    RISK_LEVELS, 
    CONFIDENCE_THRESHOLDS,
    get_event_name,
    get_display_name
)


@dataclass
class SmartDetectionResult:
    """Smart detection with learning"""
    # Primary detection (YOLO)
    primary_threat: str
    threat_level: str
    confidence: float
    
    # YOLO detections
    yolo_detections: List[Dict]
    
    # Learning insights (from other models)
    learning_insights: Dict
    boosted_confidence: float  # Confidence after model boost
    
    # Metadata
    timestamp: float
    frame_number: int
    models_consulted: List[str]
    
    # Alert
    should_alert: bool
    alert_message: str


class SmartFusionEngine:
    """
    Smart fusion that uses models for learning, not interference.
    
    - YOLO: Fast primary detector (every frame)
    - Others: Periodic validation & learning (boost accuracy)
    """
    
    def __init__(
        self, 
        learning_interval: int = 10,  # Run learning models every N frames
        boost_threshold: float = 0.6   # Below this, consult other models
    ):
        print("\n" + "="*70)
        print("🧠 INITIALIZING SMART FUSION ENGINE (Learning Mode)")
        print("="*70)
        
        # Initialize models
        self.yolo = self._init_yolo()
        self.emotion = self._init_emotion()
        self.action = self._init_action()
        self.svm = self._init_svm()
        self.audio = self._init_audio()
        
        # Smart detection settings
        self.learning_interval = learning_interval
        self.boost_threshold = boost_threshold
        
        # Frame tracking
        self.frame_count = 0
        self.detection_history = deque(maxlen=100)
        
        # Learning patterns (models learn from data)
        self.learned_patterns = {
            'emotion_correlations': {},  # e.g., fear → fire
            'action_correlations': {},    # e.g., running → emergency
            'svm_anomalies': [],          # Unusual scenes
            'confidence_boosts': {}       # Historical confidence improvements
        }
        
        # Background learning thread
        self.learning_queue = deque(maxlen=50)
        self.learning_active = True
        self.learning_thread = threading.Thread(target=self._background_learner, daemon=True)
        self.learning_thread.start()
        
        print("✅ Smart Fusion Engine Ready")
        print(f"   Primary: YOLO (every frame, ~30 FPS)")
        print(f"   Learning: Other models (every {learning_interval} frames)")
        print(f"   Boost: When YOLO confidence < {boost_threshold*100}%")
        print("="*70 + "\n")
    
    def _init_yolo(self) -> YOLOWrapper:
        """Initialize YOLO - primary detector"""
        try:
            print("⚡ Loading YOLO (Primary Detector)...")
            model_path = 'models/best.pt' if os.path.exists('models/best.pt') else 'yolov8n.pt'
            yolo = YOLOWrapper(weights=model_path, auto_download=True)
            print(f"   ✅ YOLO ready - {model_path}")
            return yolo
        except Exception as e:
            print(f"   ❌ YOLO init failed: {e}")
            return YOLOWrapper(auto_download=True)
    
    def _init_emotion(self) -> EmotionWrapper:
        """Initialize emotion - learning model"""
        try:
            print("📚 Loading Emotion Model (Learning)...")
            emotion = EmotionWrapper(path='models/emotion_model.h5')
            if emotion.model:
                print("   ✅ Emotion ready (periodic learning)")
            return emotion
        except Exception as e:
            return EmotionWrapper()
    
    def _init_action(self) -> ActionWrapper:
        """Initialize action - learning model"""
        try:
            print("📚 Loading Action Model (Learning)...")
            action = ActionWrapper(path='models/action_model.pth')
            print("   ✅ Action ready (periodic learning)")
            return action
        except Exception as e:
            return ActionWrapper()
    
    def _init_svm(self) -> SVMWrapper:
        """Initialize SVM - learning model"""
        try:
            print("📚 Loading SVM Model (Learning)...")
            svm = SVMWrapper(path='models/svm_model.pkl')
            if svm.model:
                print("   ✅ SVM ready (periodic learning)")
            return svm
        except Exception as e:
            return SVMWrapper()
    
    def _init_audio(self) -> AudioWrapper:
        """Initialize audio - learning model"""
        try:
            print("📚 Loading Audio Model (Learning)...")
            audio = AudioWrapper(path='models/audio_model.pth')
            print("   ✅ Audio ready (optional)")
            return audio
        except Exception as e:
            return AudioWrapper()
    
    def analyze_frame(self, frame: np.ndarray, audio_data: np.ndarray = None) -> SmartDetectionResult:
        """
        Smart frame analysis:
        1. Always run YOLO (fast)
        2. Periodically run other models (learning)
        3. Boost confidence when needed
        """
        self.frame_count += 1
        timestamp = time.time()
        
        # === STEP 1: YOLO PRIMARY DETECTION (Every Frame) ===
        yolo_detections = self._run_yolo_fast(frame)
        
        # Determine primary threat from YOLO
        if yolo_detections:
            top_detection = max(yolo_detections, key=lambda d: d['confidence'])
            primary_threat = top_detection['event']
            yolo_confidence = top_detection['confidence']
            threat_level = top_detection['risk_level']
        else:
            primary_threat = 'normal'
            yolo_confidence = 0.9
            threat_level = 'LOW'
        
        # === STEP 2: LEARNING MODELS (Periodic) ===
        learning_insights = {}
        models_consulted = ['YOLO']
        boosted_confidence = yolo_confidence
        
        # Run learning models periodically OR when YOLO is uncertain
        should_learn = (
            self.frame_count % self.learning_interval == 0 or 
            yolo_confidence < self.boost_threshold
        )
        
        if should_learn:
            # Queue frame for background learning
            self.learning_queue.append({
                'frame': frame.copy(),
                'timestamp': timestamp,
                'yolo_result': {
                    'threat': primary_threat,
                    'confidence': yolo_confidence
                }
            })
            
            # If YOLO uncertain, consult models NOW (not background)
            if yolo_confidence < self.boost_threshold:
                learning_insights = self._quick_consult_models(frame)
                models_consulted.extend(learning_insights.get('models_used', []))
                
                # Boost confidence based on model agreement
                boosted_confidence = self._boost_confidence(
                    yolo_confidence,
                    primary_threat,
                    learning_insights
                )
        
        # === STEP 3: DECISION & ALERT ===
        final_confidence = boosted_confidence
        should_alert = (
            (threat_level in ['CRITICAL', 'HIGH'] and final_confidence > 0.7) or
            (threat_level == 'MEDIUM' and final_confidence > 0.8)
        )
        
        alert_message = ''
        if should_alert:
            alert_message = f"{threat_level} ALERT: {primary_threat.replace('_', ' ').title()} detected ({final_confidence*100:.1f}% confidence)"
            if boosted_confidence > yolo_confidence:
                alert_message += f" [Boosted by {', '.join(models_consulted[1:])}]"
        
        result = SmartDetectionResult(
            primary_threat=primary_threat,
            threat_level=threat_level,
            confidence=final_confidence,
            yolo_detections=yolo_detections,
            learning_insights=learning_insights,
            boosted_confidence=boosted_confidence,
            timestamp=timestamp,
            frame_number=self.frame_count,
            models_consulted=models_consulted,
            should_alert=should_alert,
            alert_message=alert_message
        )
        
        self.detection_history.append(result)
        return result
    
    def _run_yolo_fast(self, frame: np.ndarray) -> List[Dict]:
        """Fast YOLO detection"""
        try:
            result = self.yolo.detect(frame)
            return result.get('objects', [])
        except Exception as e:
            return []
    
    def _quick_consult_models(self, frame: np.ndarray) -> Dict:
        """Quick consultation when YOLO uncertain"""
        insights = {
            'models_used': [],
            'emotion': None,
            'action': None,
            'svm': None,
            'agreement_score': 0.0
        }
        
        try:
            # Emotion
            if self.emotion.model:
                emotion, conf = self.emotion.predict_from_face(frame)
                insights['emotion'] = {'label': emotion, 'confidence': conf}
                insights['models_used'].append('Emotion')
            
            # Action
            if self.action.model:
                action, conf = self.action.predict(frame)
                insights['action'] = {'label': action, 'confidence': conf}
                insights['models_used'].append('Action')
            
            # SVM
            if self.svm.model:
                svm_label, conf = self.svm.predict(frame)
                insights['svm'] = {'label': svm_label, 'confidence': conf}
                insights['models_used'].append('SVM')
        
        except Exception as e:
            print(f"Model consultation error: {e}")
        
        return insights
    
    def _boost_confidence(self, yolo_conf: float, threat: str, insights: Dict) -> float:
        """Boost YOLO confidence based on other models"""
        if not insights:
            return yolo_conf
        
        boost_factor = 0.0
        
        # Check if other models agree
        if insights.get('action'):
            action_label = insights['action']['label']
            action_conf = insights['action']['confidence']
            
            # If action model agrees (e.g., both detect "fire")
            if action_label == threat and action_conf > 0.5:
                boost_factor += 0.15
        
        if insights.get('svm'):
            svm_label = insights['svm']['label']
            svm_conf = insights['svm']['confidence']
            
            # If SVM detects anomaly when YOLO detects threat
            if threat != 'normal' and svm_label == 'anomaly' and svm_conf > 0.5:
                boost_factor += 0.10
        
        # Apply boost (max +25%)
        boosted = min(yolo_conf + boost_factor, 1.0)
        
        # Learn this boost for future
        if boost_factor > 0:
            key = f"{threat}_{','.join(insights.get('models_used', []))}"
            self.learned_patterns['confidence_boosts'][key] = boost_factor
        
        return boosted
    
    def _background_learner(self):
        """Background thread that learns from queued frames"""
        print("🎓 Background learner started")
        
        while self.learning_active:
            if self.learning_queue:
                data = self.learning_queue.popleft()
                frame = data['frame']
                yolo_result = data['yolo_result']
                
                try:
                    # Run all learning models
                    if self.emotion.model:
                        emotion, conf = self.emotion.predict_from_face(frame)
                        # Learn correlation: emotion → threat
                        threat = yolo_result['threat']
                        if threat not in self.learned_patterns['emotion_correlations']:
                            self.learned_patterns['emotion_correlations'][threat] = []
                        self.learned_patterns['emotion_correlations'][threat].append(emotion)
                    
                    if self.action.model:
                        action, conf = self.action.predict(frame)
                        # Learn correlation: action → threat
                        if threat not in self.learned_patterns['action_correlations']:
                            self.learned_patterns['action_correlations'][threat] = []
                        self.learned_patterns['action_correlations'][threat].append(action)
                    
                    if self.svm.model:
                        svm_label, conf = self.svm.predict(frame)
                        # Track anomalies
                        if svm_label == 'anomaly':
                            self.learned_patterns['svm_anomalies'].append({
                                'timestamp': data['timestamp'],
                                'yolo_threat': threat
                            })
                
                except Exception as e:
                    pass  # Silent fail in background
            
            time.sleep(0.1)  # Don't hog CPU
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about what the system has learned"""
        return {
            'total_frames_processed': self.frame_count,
            'frames_in_learning_queue': len(self.learning_queue),
            'learned_patterns': {
                'emotion_threat_correlations': {
                    k: max(set(v), key=v.count) if v else 'none'
                    for k, v in self.learned_patterns['emotion_correlations'].items()
                },
                'action_threat_correlations': {
                    k: max(set(v), key=v.count) if v else 'none'
                    for k, v in self.learned_patterns['action_correlations'].items()
                },
                'total_anomalies_detected': len(self.learned_patterns['svm_anomalies']),
                'confidence_boost_patterns': len(self.learned_patterns['confidence_boosts'])
            }
        }
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'mode': 'Smart Learning Mode',
            'primary_detector': 'YOLO (every frame)',
            'learning_models': ['Emotion', 'Action', 'SVM', 'Audio'],
            'learning_interval': f'Every {self.learning_interval} frames',
            'boost_threshold': f'{self.boost_threshold*100}%',
            'frames_processed': self.frame_count,
            'learning_stats': self.get_learning_stats()
        }
    
    def shutdown(self):
        """Clean shutdown"""
        print("🛑 Shutting down background learner...")
        self.learning_active = False
        if self.learning_thread.is_alive():
            self.learning_thread.join(timeout=2)
        print("✅ Shutdown complete")
