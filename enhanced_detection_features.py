"""
Enhanced Detection System with Advanced Features
- Event Sequencing (explosion → smoke → fire transitions)
- Temporal Logic (cooldown periods, event transitions)
- Feature Engineering (weighted features, motion history, brightness history)
- Pattern Enhancement (targeted learning for low recall events)
"""

import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta

class TemporalEventTracker:
    """Track event sequences and transitions over time"""
    
    def __init__(self):
        # Event history tracking
        self.event_history = deque(maxlen=100)  # Last 100 events
        self.last_event_time = {}  # timestamp of last detection per event
        self.event_transitions = defaultdict(int)  # Track common transitions
        
        # Cooldown periods (frames) to avoid repeated alerts
        self.cooldown_periods = {
            'explosion': 30,  # 30 frames (~1 second at 30fps)
            'fire': 20,
            'vehicle_accident': 40,
            'fighting': 25,
            'smoke': 15
        }
        
        # Expected event sequences
        self.event_sequences = {
            'explosion': ['smoke', 'fire'],  # After explosion, expect smoke then fire
            'fire': ['smoke'],  # Fire produces smoke
            'vehicle_accident': ['smoke', 'fire'],  # Accidents can lead to fire
            'smoke': []  # Smoke doesn't lead to other events
        }
        
        # Sequence boost factors
        self.sequence_boost = 20  # Boost confidence by 20% if following expected sequence
    
    def record_event(self, event_type, frame_num, confidence):
        """Record detected event with timestamp"""
        current_time = datetime.now()
        
        # Check if this is a valid transition from previous event
        is_valid_sequence = False
        if len(self.event_history) > 0:
            last_event = self.event_history[-1]
            if event_type in self.event_sequences.get(last_event['type'], []):
                is_valid_sequence = True
                transition_key = f"{last_event['type']}_to_{event_type}"
                self.event_transitions[transition_key] += 1
        
        # Record event
        event_record = {
            'type': event_type,
            'frame': frame_num,
            'confidence': confidence,
            'timestamp': current_time,
            'is_sequence': is_valid_sequence
        }
        self.event_history.append(event_record)
        self.last_event_time[event_type] = frame_num
        
        return is_valid_sequence
    
    def check_cooldown(self, event_type, current_frame):
        """Check if event is in cooldown period"""
        if event_type not in self.last_event_time:
            return False  # Never detected before
        
        last_frame = self.last_event_time[event_type]
        cooldown = self.cooldown_periods.get(event_type, 20)
        
        return (current_frame - last_frame) < cooldown
    
    def get_expected_events(self):
        """Get list of events expected based on recent history"""
        if len(self.event_history) == 0:
            return []
        
        last_event = self.event_history[-1]
        return self.event_sequences.get(last_event['type'], [])
    
    def should_boost_event(self, event_type):
        """Check if event should get sequence boost"""
        expected = self.get_expected_events()
        return event_type in expected
    
    def get_sequence_stats(self):
        """Get statistics about event transitions"""
        return dict(self.event_transitions)


class FeatureEngineer:
    """Advanced feature engineering for event detection"""
    
    def __init__(self, history_size=30):
        # Motion history tracking
        self.motion_history = deque(maxlen=history_size)
        self.brightness_history = deque(maxlen=history_size)
        self.edge_history = deque(maxlen=history_size)
        self.color_history = deque(maxlen=history_size)
        
        # Feature weights per event type
        self.feature_weights = {
            'explosion': {
                'brightness_spike': 0.30,  # Most important
                'motion_spike': 0.25,
                'smoke': 0.20,
                'edges': 0.15,
                'debris': 0.10
            },
            'fire': {
                'flame_colors': 0.35,  # Most important
                'brightness': 0.25,
                'smoke': 0.20,
                'flickering': 0.15,
                'heat_signature': 0.05
            },
            'vehicle_accident': {
                'edges': 0.30,  # Most important (debris/damage)
                'motion_spike': 0.25,
                'metal_colors': 0.20,
                'impact_pattern': 0.15,
                'smoke': 0.10
            },
            'fighting': {
                'motion': 0.35,  # Most important
                'edges': 0.25,
                'sustained_activity': 0.20,
                'color_chaos': 0.15,
                'spatial_changes': 0.05
            },
            'smoke': {
                'smoke_coverage': 0.40,  # Most important
                'smoke_density': 0.30,
                'no_flash': 0.20,  # Standalone smoke has no flash
                'texture': 0.10
            }
        }
    
    def update_history(self, motion, brightness, edges, colors):
        """Update feature history"""
        self.motion_history.append(motion)
        self.brightness_history.append(brightness)
        self.edge_history.append(edges)
        self.color_history.append(colors)
    
    def detect_brightness_spike(self, current_brightness, threshold=30):
        """Detect sudden brightness spike (explosion flash)"""
        if len(self.brightness_history) < 3:
            return False, 0
        
        recent_avg = np.mean(list(self.brightness_history)[-3:])
        spike = current_brightness - recent_avg
        
        return spike > threshold, spike
    
    def detect_motion_spike(self, current_motion, multiplier=2.0):
        """Detect sudden motion spike"""
        if len(self.motion_history) < 5:
            return False, 0
        
        recent_avg = np.mean(list(self.motion_history)[-5:])
        if recent_avg == 0:
            return False, 0
        
        ratio = current_motion / recent_avg
        return ratio > multiplier, ratio
    
    def detect_flickering(self, window=10):
        """Detect brightness flickering (fire characteristic)"""
        if len(self.brightness_history) < window:
            return False, 0
        
        recent = list(self.brightness_history)[-window:]
        variance = np.var(recent)
        
        return variance > 50, variance
    
    def detect_sustained_activity(self, threshold=50, duration=15):
        """Detect sustained motion (fighting characteristic)"""
        if len(self.motion_history) < duration:
            return False, 0
        
        recent = list(self.motion_history)[-duration:]
        above_threshold = sum(1 for m in recent if m > threshold)
        
        ratio = above_threshold / duration
        return ratio > 0.7, ratio
    
    def compute_weighted_score(self, event_type, features):
        """Compute weighted score based on feature importance"""
        weights = self.feature_weights.get(event_type, {})
        
        total_score = 0
        for feature_name, feature_value in features.items():
            weight = weights.get(feature_name, 0)
            total_score += feature_value * weight * 100  # Scale to 0-100
        
        return min(total_score, 100)


class PatternTargetLearner:
    """Target pattern learning for events with low recall"""
    
    def __init__(self):
        self.low_recall_events = []
        self.recommended_patterns = {}
    
    def analyze_performance(self, test_results):
        """Analyze test results to identify low recall events"""
        self.low_recall_events = []
        
        for result in test_results:
            if result['recall'] < 0.7:  # Less than 70% recall
                self.low_recall_events.append({
                    'event': result['event_type'],
                    'recall': result['recall'],
                    'precision': result['precision']
                })
        
        return self.low_recall_events
    
    def generate_learning_plan(self, target_patterns=100000):
        """Generate targeted learning plan"""
        if not self.low_recall_events:
            return None
        
        plan = {
            'total_patterns': target_patterns,
            'per_event': {}
        }
        
        # Distribute patterns based on recall deficit
        total_deficit = sum(1 - evt['recall'] for evt in self.low_recall_events)
        
        for evt in self.low_recall_events:
            deficit = 1 - evt['recall']
            allocation = int((deficit / total_deficit) * target_patterns)
            
            plan['per_event'][evt['event']] = {
                'patterns_to_learn': allocation,
                'current_recall': evt['recall'],
                'target_recall': 0.85,
                'search_queries': self.get_search_queries(evt['event'])
            }
        
        return plan
    
    def get_search_queries(self, event_type):
        """Get focused search queries for specific event type"""
        focused_queries = {
            'explosion': [
                'explosion bright flash', 'explosion debris cloud', 'explosive detonation',
                'bomb blast moment', 'explosion shockwave', 'explosion impact frame'
            ],
            'fire': [
                'fire flames closeup', 'building fire interior', 'fire spreading',
                'flames burning bright', 'fire engulfing', 'intense fire blaze'
            ],
            'vehicle_accident': [
                'car crash impact', 'vehicle collision damage', 'car wreck debris',
                'accident scene damage', 'collision broken glass', 'car crash metal'
            ],
            'fighting': [
                'people fighting action', 'physical fight motion', 'brawl fighting',
                'street fight action', 'fighting movement', 'combat action'
            ],
            'smoke': [
                'thick smoke cloud', 'dense smoke billowing', 'heavy smoke plume',
                'dark smoke rising', 'white smoke cloud', 'smoke spreading'
            ]
        }
        
        return focused_queries.get(event_type, [])


# Export utility function
def get_enhancement_recommendations(test_results):
    """Get specific enhancement recommendations based on test results"""
    
    learner = PatternTargetLearner()
    low_recall = learner.analyze_performance(test_results)
    
    recommendations = {
        'temporal_logic': {
            'enabled': True,
            'cooldown_tuning': 'Adjust cooldown periods based on false positive rate',
            'sequence_tracking': 'Enable explosion→smoke→fire sequence detection'
        },
        'feature_engineering': {
            'enabled': True,
            'history_size': 30,
            'weighted_scoring': 'Use event-specific feature weights',
            'spike_detection': 'Enable brightness and motion spike detection'
        },
        'pattern_learning': learner.generate_learning_plan(100000)
    }
    
    return recommendations
