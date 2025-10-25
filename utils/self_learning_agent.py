"""
SELF-LEARNING INTELLIGENT AGENT
================================
This agent LEARNS while it works (online learning):
✅ Starts with YOLO detections
✅ Learns patterns from videos in real-time
✅ Improves accuracy over time
✅ No pre-training needed - learns from your data

Learning Strategy:
1. Use YOLO for initial detection
2. Extract features from detected events
3. Build knowledge base of event patterns
4. Use learned patterns to boost future detections
"""

import cv2
import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from utils.yolo_wrapper import YOLOWrapper
from utils.emotion_wrapper import EmotionWrapper
from utils.action_wrapper import ActionWrapper
from models.class_mapping import DISPLAY_NAMES, RISK_LEVELS


@dataclass
class LearningRecord:
    """Record of learned event"""
    event_type: str
    features: np.ndarray
    confidence: float
    timestamp: float
    frame_number: int
    emotion: Optional[str] = None
    action: Optional[str] = None


class SelfLearningAgent:
    """
    Agent that learns from videos as it processes them
    No pre-training needed - learns on the fly
    """
    
    def __init__(self, knowledge_base_path="models/agent_knowledge.pkl"):
        print("\n" + "="*70)
        print("🧠 INITIALIZING SELF-LEARNING AGENT")
        print("="*70 + "\n")
        
        self.knowledge_base_path = knowledge_base_path
        
        # Core models
        print("⚡ Loading Detection Models...")
        self.yolo = YOLOWrapper(auto_download=True)
        print("   ✅ YOLO ready")
        
        self.emotion = EmotionWrapper()
        print("   ✅ Emotion ready")
        
        self.action = ActionWrapper()
        print("   ✅ Action ready\n")
        
        # Learning system
        print("🎓 Initializing Learning System...")
        
        # Knowledge base: stores learned patterns for each event type
        self.knowledge_base = {
            'explosion': [],
            'fighting': [],
            'fire': [],
            'vehicle_accident': [],
            'normal': []
        }
        
        # Pattern memory (keep last N examples per class)
        self.memory_size = 100  # Keep 100 examples per event type
        
        # Learning statistics
        self.learning_stats = {
            'total_frames': 0,
            'events_learned': 0,
            'explosions_learned': 0,
            'fighting_learned': 0,
            'fire_learned': 0,
            'accidents_learned': 0,
            'patterns_used': 0,
            'accuracy_boost': 0
        }
        
        # Load existing knowledge if available
        self._load_knowledge()
        
        print(f"   ✅ Learning system ready")
        print(f"   📚 Knowledge loaded: {sum(len(v) for v in self.knowledge_base.values())} patterns\n")
        
        print("="*70)
        print("✅ SELF-LEARNING AGENT READY")
        print("="*70)
        print("\n💡 Learning Mode: ACTIVE")
        print("   - Learns from every detection")
        print("   - Improves accuracy over time")
        print("   - Saves knowledge automatically")
        print("="*70 + "\n")
    
    def _extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from frame for learning
        Simple but effective features
        """
        try:
            # Resize for consistency
            frame_resized = cv2.resize(frame, (128, 128))
            
            # Color histogram
            hist_r = cv2.calcHist([frame_resized], [0], None, [16], [0, 256]).flatten()
            hist_g = cv2.calcHist([frame_resized], [1], None, [16], [0, 256]).flatten()
            hist_b = cv2.calcHist([frame_resized], [2], None, [16], [0, 256]).flatten()
            
            # Edge detection
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256]).flatten()
            
            # Brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Combine features
            features = np.concatenate([hist_r, hist_g, hist_b, edge_hist, [brightness, contrast]])
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
        except:
            return None
    
    def _similarity_score(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate similarity between two feature vectors
        Returns: 0-1 (1 = identical, 0 = completely different)
        """
        return np.dot(features1, features2)  # Cosine similarity (already normalized)
    
    def _learn_from_detection(self, frame: np.ndarray, event_type: str, confidence: float, 
                              frame_number: int, emotion: str = None, action: str = None):
        """
        Learn from a detected event
        Adds pattern to knowledge base
        """
        # Extract features
        features = self._extract_features(frame)
        if features is None:
            return
        
        # Create learning record
        record = LearningRecord(
            event_type=event_type,
            features=features,
            confidence=confidence,
            timestamp=time.time(),
            frame_number=frame_number,
            emotion=emotion,
            action=action
        )
        
        # Add to knowledge base
        self.knowledge_base[event_type].append(record)
        
        # Limit memory size (keep only recent patterns)
        if len(self.knowledge_base[event_type]) > self.memory_size:
            self.knowledge_base[event_type].pop(0)  # Remove oldest
        
        # Update statistics
        self.learning_stats['events_learned'] += 1
        if event_type == 'explosion':
            self.learning_stats['explosions_learned'] += 1
        elif event_type == 'fighting':
            self.learning_stats['fighting_learned'] += 1
        elif event_type == 'fire':
            self.learning_stats['fire_learned'] += 1
        elif event_type == 'vehicle_accident':
            self.learning_stats['accidents_learned'] += 1
    
    def _check_learned_patterns(self, frame: np.ndarray, event_type: str) -> float:
        """
        Check if current frame matches learned patterns
        Returns: confidence boost (0-0.3)
        """
        if event_type not in self.knowledge_base:
            return 0.0
        
        learned_patterns = self.knowledge_base[event_type]
        if len(learned_patterns) == 0:
            return 0.0
        
        # Extract features from current frame
        current_features = self._extract_features(frame)
        if current_features is None:
            return 0.0
        
        # Compare with learned patterns
        similarities = []
        for pattern in learned_patterns:
            similarity = self._similarity_score(current_features, pattern.features)
            similarities.append(similarity)
        
        # Use average of top 5 matches
        top_similarities = sorted(similarities, reverse=True)[:5]
        avg_similarity = np.mean(top_similarities) if top_similarities else 0.0
        
        # Convert to confidence boost (0-0.3)
        boost = avg_similarity * 0.3
        
        if boost > 0.1:
            self.learning_stats['patterns_used'] += 1
        
        return boost
    
    def detect(self, frame: np.ndarray, frame_number: int = 0) -> Dict:
        """
        Detect events and learn from them
        """
        self.learning_stats['total_frames'] += 1
        
        # STEP 1: YOLO Detection
        yolo_results = self.yolo.detect(frame)
        detections = yolo_results.get('detections', [])
        
        # Find best detection
        best_event = None
        max_confidence = 0.0
        
        for detection in detections:
            event_name = detection.get('event', '').lower()
            confidence = detection.get('confidence', 0)
            
            # Lower threshold to catch more events
            if event_name in ['explosion', 'fighting', 'fire', 'vehicle_accident'] and confidence > 0.25:
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_event = {
                        'event': event_name,
                        'confidence': confidence,
                        'class_id': detection.get('class_id', -1),
                        'bbox': detection.get('bbox', [])
                    }
        
        if not best_event:
            best_event = {
                'event': 'normal',
                'confidence': 0.0,
                'class_id': -1,
                'bbox': []
            }
        
        event_type = best_event['event']
        base_confidence = max_confidence
        
        # STEP 2: Check learned patterns
        pattern_boost = 0.0
        if event_type != 'normal':
            pattern_boost = self._check_learned_patterns(frame, event_type)
        
        # STEP 3: Support models (for specific events)
        emotion_result = None
        action_result = None
        support_boost = 0.0
        
        if event_type == 'fighting' and base_confidence > 0.3:
            try:
                emotion_data = self.emotion.analyze(frame)
                if emotion_data:
                    emotion_result = emotion_data.get('dominant_emotion', 'neutral')
                    if emotion_result in ['angry', 'fear', 'disgust']:
                        support_boost += 0.1
            except:
                pass
            
            try:
                action_data = self.action.classify(frame)
                if action_data:
                    action_result = action_data.get('action', 'unknown')
                    if action_result == 'Fighting':
                        support_boost += 0.15
            except:
                pass
        
        # STEP 4: Calculate final confidence
        final_confidence = min(base_confidence + pattern_boost + support_boost, 1.0)
        
        # STEP 5: Learn from this detection (if confident enough)
        if event_type != 'normal' and final_confidence > 0.4:  # Lower threshold for learning
            self._learn_from_detection(
                frame, event_type, final_confidence, 
                frame_number, emotion_result, action_result
            )
        
        # Track accuracy improvement
        if pattern_boost > 0 or support_boost > 0:
            self.learning_stats['accuracy_boost'] += (pattern_boost + support_boost)
        
        # Prepare result
        display_name = DISPLAY_NAMES.get(best_event['class_id'], event_type.title())
        risk_level = RISK_LEVELS.get(event_type, 'LOW')
        should_alert = (event_type != 'normal' and final_confidence > 0.5)
        
        return {
            'event_type': event_type,
            'display_name': display_name,
            'risk_level': risk_level,
            'base_confidence': base_confidence,
            'pattern_boost': pattern_boost,
            'support_boost': support_boost,
            'final_confidence': final_confidence,
            'detections': detections,
            'emotion': emotion_result,
            'action': action_result,
            'should_alert': should_alert,
            'frame_number': frame_number,
            'learned_patterns': len(self.knowledge_base[event_type])
        }
    
    def save_knowledge(self):
        """Save learned knowledge to disk"""
        try:
            # Convert to serializable format
            knowledge_data = {}
            for event_type, records in self.knowledge_base.items():
                knowledge_data[event_type] = [
                    {
                        'event_type': r.event_type,
                        'features': r.features.tolist(),
                        'confidence': r.confidence,
                        'timestamp': r.timestamp,
                        'frame_number': r.frame_number,
                        'emotion': r.emotion,
                        'action': r.action
                    }
                    for r in records
                ]
            
            save_data = {
                'knowledge_base': knowledge_data,
                'learning_stats': self.learning_stats,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.knowledge_base_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            return True
        except Exception as e:
            print(f"⚠️  Warning: Could not save knowledge: {e}")
            return False
    
    def _load_knowledge(self):
        """Load previously learned knowledge"""
        try:
            if Path(self.knowledge_base_path).exists():
                with open(self.knowledge_base_path, 'rb') as f:
                    save_data = pickle.load(f)
                
                # Restore knowledge base
                knowledge_data = save_data.get('knowledge_base', {})
                for event_type, records in knowledge_data.items():
                    self.knowledge_base[event_type] = [
                        LearningRecord(
                            event_type=r['event_type'],
                            features=np.array(r['features']),
                            confidence=r['confidence'],
                            timestamp=r['timestamp'],
                            frame_number=r['frame_number'],
                            emotion=r.get('emotion'),
                            action=r.get('action')
                        )
                        for r in records
                    ]
                
                # Restore stats
                self.learning_stats.update(save_data.get('learning_stats', {}))
                
                return True
        except Exception as e:
            print(f"   ℹ️  No previous knowledge found (starting fresh)")
            return False
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        total_learned = sum(len(v) for v in self.knowledge_base.values())
        
        return {
            **self.learning_stats,
            'total_patterns': total_learned,
            'explosion_patterns': len(self.knowledge_base['explosion']),
            'fighting_patterns': len(self.knowledge_base['fighting']),
            'fire_patterns': len(self.knowledge_base['fire']),
            'accident_patterns': len(self.knowledge_base['vehicle_accident']),
            'avg_boost_per_frame': self.learning_stats['accuracy_boost'] / max(self.learning_stats['patterns_used'], 1)
        }


if __name__ == "__main__":
    print("Self-Learning Agent Module")
    print("Use test_self_learning_agent.py to run tests")
