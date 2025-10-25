"""
PURE AGENT DETECTOR
===================
Detects events using ONLY learned patterns
NO YOLO - Pure pattern matching
"""

import numpy as np
import cv2
import pickle
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PureAgentResult:
    """Detection result from pure agent"""
    event_type: str
    confidence: float
    top_matches: List[Tuple[str, float]]  # [(event, confidence), ...]
    pattern_count: int
    detection_time_ms: float
    should_alert: bool
    risk_level: str

class PureAgentDetector:
    """
    Pure pattern-based detector
    NO YOLO - Uses only learned patterns from web
    """
    
    def __init__(self, knowledge_path: str = "models/pure_agent_knowledge.pkl"):
        print("\n" + "="*70)
        print("🤖 INITIALIZING PURE AGENT DETECTOR")
        print("="*70)
        print("🚫 NO YOLO - Pure pattern-based detection")
        
        # Load knowledge
        print(f"📚 Loading knowledge: {knowledge_path}")
        with open(knowledge_path, 'rb') as f:
            knowledge = pickle.load(f)
        
        self.patterns = knowledge['patterns']
        self.characteristics = knowledge['characteristics']
        self.total_patterns = knowledge['total_learned']
        
        print(f"✅ Knowledge loaded:")
        print(f"   📊 Total patterns: {self.total_patterns:,}")
        print(f"   🎯 Event types: {len(self.patterns)}")
        for event, patterns in self.patterns.items():
            print(f"      - {event}: {len(patterns):,} patterns")
        
        # Risk levels
        self.risk_levels = {
            'explosion': 'CRITICAL',
            'fire': 'CRITICAL',
            'fighting': 'HIGH',
            'vehicle_accident': 'HIGH',
            'normal': 'LOW'
        }
        
        # Display names
        self.display_names = {
            'explosion': 'Explosion Detected',
            'fire': 'Fire Detected',
            'fighting': 'Violence Detected',
            'vehicle_accident': 'Vehicle Accident',
            'normal': 'Normal'
        }
        
        print(f"✅ Pure Agent Ready")
        print("="*70 + "\n")
    
    def extract_features(self, frame: np.ndarray) -> Dict:
        """
        Extract same 256+ features used during learning
        """
        features = {}
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        h, w = gray.shape
        
        # 1. BRIGHTNESS FEATURES
        features['brightness_mean'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['brightness_max'] = np.max(gray)
        features['brightness_min'] = np.min(gray)
        features['brightness_median'] = np.median(gray)
        
        # Brightness in regions (3x3 grid)
        regions_h = h // 3
        regions_w = w // 3
        for i in range(3):
            for j in range(3):
                region = gray[i*regions_h:(i+1)*regions_h, j*regions_w:(j+1)*regions_w]
                features[f'brightness_region_{i}_{j}'] = np.mean(region)
        
        # Brightness histogram
        hist, _ = np.histogram(gray, bins=16, range=(0, 256))
        for i, val in enumerate(hist):
            features[f'brightness_hist_{i}'] = val / (h * w)
        
        # 2. EDGE FEATURES
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (h * w)
        features['edge_mean'] = np.mean(edges)
        features['edge_std'] = np.std(edges)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['edge_horizontal'] = np.mean(np.abs(sobel_x))
        features['edge_vertical'] = np.mean(np.abs(sobel_y))
        features['edge_magnitude'] = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
        
        # Edge regions
        for i in range(3):
            for j in range(3):
                region = edges[i*regions_h:(i+1)*regions_h, j*regions_w:(j+1)*regions_w]
                features[f'edge_region_{i}_{j}'] = np.sum(region > 0) / (regions_h * regions_w)
        
        # 3. TEXTURE FEATURES
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['texture_variance'] = np.var(laplacian)
        features['texture_mean'] = np.mean(np.abs(laplacian))
        features['texture_std'] = np.std(laplacian)
        
        # Texture in regions (4x4 grid)
        for i in range(4):
            for j in range(4):
                reg_h = h // 4
                reg_w = w // 4
                region = gray[i*reg_h:(i+1)*reg_h, j*reg_w:(j+1)*reg_w]
                lap_region = cv2.Laplacian(region, cv2.CV_64F)
                features[f'texture_region_{i}_{j}'] = np.var(lap_region)
        
        # 4. GRADIENT FEATURES
        grad_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['gradient_mean'] = np.mean(grad_magnitude)
        features['gradient_std'] = np.std(grad_magnitude)
        features['gradient_max'] = np.max(grad_magnitude)
        
        # 5. STATISTICAL FEATURES
        hist_normalized = hist / (h * w + 1e-10)
        features['entropy'] = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        features['contrast'] = np.max(gray) - np.min(gray)
        features['energy'] = np.sum(gray ** 2) / (h * w)
        
        return features
    
    def calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Calculate similarity between two feature sets
        Using normalized Euclidean distance
        """
        # Get common features
        common_keys = set(features1.keys()) & set(features2.keys())
        
        if not common_keys:
            return 0.0
        
        # Calculate distance
        total_dist = 0.0
        for key in common_keys:
            val1 = features1[key]
            val2 = features2[key]
            
            # Normalize to prevent one feature dominating
            if key.startswith('brightness'):
                diff = abs(val1 - val2) / 255.0
            elif key.startswith('edge') or key.startswith('gradient'):
                diff = abs(val1 - val2) / max(abs(val1) + abs(val2), 1.0)
            else:
                diff = abs(val1 - val2) / max(abs(val1) + abs(val2), 1.0)
            
            total_dist += diff ** 2
        
        # Normalized distance
        distance = np.sqrt(total_dist / len(common_keys))
        
        # Convert to similarity (0-1)
        similarity = 1.0 / (1.0 + distance)
        
        return similarity
    
    def detect(self, frame: np.ndarray, top_k: int = 100) -> PureAgentResult:
        """
        Detect event using pure pattern matching
        NO YOLO involved
        """
        import time
        start_time = time.time()
        
        # Extract features from frame
        frame_features = self.extract_features(frame)
        
        # Match against all patterns
        event_scores = {}
        
        for event_type, patterns in self.patterns.items():
            # Sample patterns (check top_k random patterns for speed)
            sample_size = min(top_k, len(patterns))
            sampled_patterns = np.random.choice(len(patterns), sample_size, replace=False)
            
            similarities = []
            for idx in sampled_patterns:
                pattern = patterns[idx]
                similarity = self.calculate_similarity(frame_features, pattern)
                similarities.append(similarity)
            
            # Average similarity for this event
            event_scores[event_type] = np.mean(similarities) if similarities else 0.0
        
        # Sort by confidence
        sorted_events = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Best match
        best_event, best_confidence = sorted_events[0]
        
        # Alert decision
        should_alert = (best_event != 'normal' and best_confidence > 0.6)
        
        detection_time = (time.time() - start_time) * 1000
        
        return PureAgentResult(
            event_type=best_event,
            confidence=best_confidence,
            top_matches=sorted_events[:3],
            pattern_count=len(self.patterns[best_event]),
            detection_time_ms=detection_time,
            should_alert=should_alert,
            risk_level=self.risk_levels.get(best_event, 'LOW')
        )
    
    def get_display_name(self, event_type: str) -> str:
        """Get display name for event"""
        return self.display_names.get(event_type, event_type.title())


if __name__ == "__main__":
    print("Pure Agent Detector Module")
    print("Use test_pure_agent.py to run tests")
