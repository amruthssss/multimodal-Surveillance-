"""
INTELLIGENT EVENT DIFFERENTIATION LOGIC
Based on visual, motion, context, and temporal cues
Implements multi-layer reasoning: Visual → Motion → Contextual → Temporal
"""

import cv2
import numpy as np

class SmartEventDifferentiator:
    """
    Multi-layer reasoning system for accurate event detection
    Layer 1: Visual Cues (brightness, color, texture)
    Layer 2: Motion Signatures (optical flow patterns)
    Layer 3: Context & Objects (YOLO detections)
    Layer 4: Temporal Consistency (duration patterns)
    """
    
    def __init__(self):
        # Event signatures for quick lookup
        self.event_signatures = {
            'explosion': {
                'visual': 'bright_flash_fireball',
                'motion': 'radial_outward',
                'color': 'white_orange_flash',
                'duration': '0.2-1.5s',
                'objects': 'fire_smoke_debris',
                'aftermath': 'smoke_persists'
            },
            'fire': {
                'visual': 'flickering_flames',
                'motion': 'upward_flicker',
                'color': 'orange_red_yellow',
                'duration': '10+s_continuous',
                'objects': 'fire_source_visible',
                'aftermath': 'slowly_increases'
            },
            'accident': {
                'visual': 'collision_debris',
                'motion': 'linear_then_stop',
                'color': 'normal_gray_tones',
                'duration': '1-3s_impact',
                'objects': 'vehicles_roads',
                'aftermath': 'static_scene'
            },
            'fighting': {
                'visual': 'human_movement',
                'motion': 'chaotic_bidirectional',
                'color': 'normal_brightness',
                'duration': '2-6s_bursts',
                'objects': 'multiple_humans',
                'aftermath': 'active_reactions'
            },
            'smoke': {
                'visual': 'cloudy_diffused',
                'motion': 'diffused_drift',
                'color': 'gray_black_low_brightness',
                'duration': '10+s_gradual',
                'objects': 'smoke_clouds',
                'aftermath': 'lingers_long'
            }
        }
        
        # Motion thresholds learned from patterns
        self.motion_thresholds = {
            'explosion': {'magnitude': 'very_high', 'pattern': 'radial', 'range': (450, 2000)},
            'accident': {'magnitude': 'medium_high', 'pattern': 'linear', 'range': (200, 450)},
            'fire': {'magnitude': 'low', 'pattern': 'upward', 'range': (20, 100)},
            'smoke': {'magnitude': 'very_low', 'pattern': 'diffused', 'range': (5, 50)},
            'fighting': {'magnitude': 'medium_high', 'pattern': 'chaotic', 'range': (60, 200)}
        }
    
    def layer1_visual_analysis(self, frame):
        """Layer 1: Visual Cues Analysis"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        results = {}
        
        # Brightness analysis
        brightness = np.mean(gray)
        brightness_spike = brightness > 200
        results['brightness'] = brightness
        results['brightness_spike'] = brightness_spike
        
        # Color analysis
        orange_mask = cv2.inRange(hsv, np.array([10, 100, 150]), np.array([30, 255, 255]))
        orange_ratio = np.count_nonzero(orange_mask) / (frame.shape[0] * frame.shape[1])
        
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 50, 180]))
        gray_ratio = np.count_nonzero(gray_mask) / (frame.shape[0] * frame.shape[1])
        
        results['orange_ratio'] = orange_ratio
        results['gray_ratio'] = gray_ratio
        
        # Texture variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()
        results['texture_variance'] = texture_variance
        
        return results
    
    def layer2_motion_analysis(self, motion_spike, flow_pattern='unknown'):
        """Layer 2: Motion Signature Analysis"""
        
        for event, thresholds in self.motion_thresholds.items():
            min_val, max_val = thresholds['range']
            if min_val <= motion_spike <= max_val:
                return event, thresholds['pattern']
        
        return 'unknown', 'unknown'
    
    def layer3_context_analysis(self, yolo_detections):
        """Layer 3: Context & Object Understanding"""
        
        context = {
            'vehicles_present': yolo_detections.get('vehicle_accident', 0) > 20,
            'fire_present': yolo_detections.get('fire', 0) > 30,
            'smoke_present': yolo_detections.get('explosion', 0) > 20,  # YOLO sometimes confuses
            'people_present': yolo_detections.get('fighting', 0) > 20
        }
        
        # Prioritization logic
        if context['fire_present'] and context['smoke_present']:
            return 'explosion_or_fire'
        elif context['vehicles_present']:
            return 'accident_likely'
        elif context['people_present']:
            return 'fighting_likely'
        elif context['smoke_present']:
            return 'smoke_likely'
        elif context['fire_present']:
            return 'fire_likely'
        
        return 'unclear'
    
    def layer4_temporal_check(self, event_duration):
        """Layer 4: Temporal Consistency"""
        
        # Duration patterns (in seconds)
        if event_duration < 1.5:
            return 'explosion_possible'  # Very short burst
        elif 1 <= event_duration <= 3:
            return 'accident_possible'   # Brief impact
        elif 2 <= event_duration <= 6:
            return 'fighting_possible'   # Medium bursts
        elif event_duration > 5:
            return 'fire_or_smoke_possible'  # Long duration
        
        return 'unclear'
    
    def resolve_event_overlap(self, candidates):
        """
        Resolve overlapping detections using priority rules:
        - Explosion + Fire + Smoke → Explosion (primary cause)
        - Fire + Smoke → Fire
        - Fire + Accident → Accident (if vehicle impact visible)
        - Fighting + Accident → Fighting (if no vehicle)
        - Explosion + Accident → Explosion (if flash/fire present)
        """
        
        if not candidates:
            return None
        
        # Priority order
        if 'explosion' in candidates and candidates['explosion'] > 70:
            return 'explosion'
        
        if 'accident' in candidates and candidates['accident'] > 65:
            if 'fire' in candidates and candidates['fire'] > 60:
                return 'accident'  # Accident with fire
            return 'accident'
        
        if 'fire' in candidates and candidates['fire'] > 65:
            return 'fire'
        
        if 'fighting' in candidates and candidates['fighting'] > 65:
            return 'fighting'
        
        if 'smoke' in candidates and candidates['smoke'] > 60:
            return 'smoke'
        
        # Return highest confidence
        return max(candidates, key=candidates.get)
    
    def full_event_analysis(self, visual_data, motion_spike, yolo_detections, duration=0):
        """
        Complete multi-layer analysis
        Returns: (event_type, confidence, reasoning)
        """
        
        # Layer 1: Visual
        if visual_data['brightness_spike'] and motion_spike > 450:
            return 'explosion', 90, 'Bright flash + radial burst'
        
        if visual_data['orange_ratio'] > 0.10 and motion_spike < 100:
            return 'fire', 80, 'Orange flames + low motion'
        
        if visual_data['gray_ratio'] > 0.30 and motion_spike < 50:
            return 'smoke', 75, 'Gray clouds + slow drift'
        
        # Layer 2: Motion
        motion_event, motion_pattern = self.layer2_motion_analysis(motion_spike)
        
        if motion_event == 'accident' and 200 <= motion_spike <= 450:
            return 'accident', 85, f'Motion spike {motion_spike:.1f} in collision range'
        
        if motion_event == 'fighting' and 60 <= motion_spike <= 200:
            return 'fighting', 80, f'Chaotic motion {motion_spike:.1f}'
        
        # Layer 3: Context
        context = self.layer3_context_analysis(yolo_detections)
        
        if context == 'accident_likely' and motion_spike > 200:
            return 'accident', 85, 'Vehicles + impact motion'
        
        if context == 'fighting_likely' and motion_spike > 60:
            return 'fighting', 80, 'Multiple people + erratic motion'
        
        # Default: unclear
        return 'unknown', 0, 'Insufficient evidence'


# Decision flow helper
def detect_event(frame, motion_spike, yolo_detections, duration=0):
    """
    Main decision flow:
    DETECT OBJECTS → ANALYZE MOTION → EVALUATE COLOR/BRIGHTNESS → APPLY TEMPORAL LOGIC → FINAL LABEL
    """
    
    differentiator = SmartEventDifferentiator()
    
    # Step 1: Visual analysis
    visual = differentiator.layer1_visual_analysis(frame)
    
    # Step 2: Full analysis
    event, confidence, reasoning = differentiator.full_event_analysis(
        visual, motion_spike, yolo_detections, duration
    )
    
    return event, confidence, reasoning
