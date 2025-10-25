"""
INTELLIGENT EVENT DIFFERENTIATION SYSTEM
Based on Google search analysis of visual, motion, and temporal signatures

Event Signatures:
- Accident: LINEAR collision, directional → stop, vehicles visible, neutral colors
- Explosion: RADIAL burst, sudden flash, white/orange, very short (0.5-1s)
- Fire: UPWARD flicker, orange/yellow, static location, long duration
- Smoke: SLOW drift, gray/black, diffused, long duration
- Fighting: IRREGULAR motion, multiple humans, back-and-forth, normal colors
"""

import cv2
import numpy as np

class SmartEventDifferentiator:
    """Differentiates between accident, explosion, fire, smoke, fighting"""
    
    def __init__(self):
        # Event-specific signatures learned from Google images/videos
        self.signatures = {
            'accident': {
                'motion_type': 'linear_then_stop',
                'motion_range': (100, 300),  # Moderate spike, not extreme
                'color_pattern': 'neutral',   # Gray, dust, no bright flash
                'duration': 'medium',         # 1-3 seconds
                'objects': ['vehicle', 'car', 'bike'],
                'no_flash': True,             # NO bright orange flash
                'directional': True           # Linear collision
            },
            'explosion': {
                'motion_type': 'radial_burst',
                'motion_range': (300, 1000),  # VERY high spike
                'color_pattern': 'flash',     # White/orange flash
                'duration': 'very_short',     # 0.5-1 second
                'objects': ['fire', 'smoke'],
                'brightness_spike': 200,      # Sudden brightness
                'radial': True                # Outward burst
            },
            'fire': {
                'motion_type': 'upward_flicker',
                'motion_range': (20, 100),    # Low, flickering motion
                'color_pattern': 'flames',    # Orange/yellow/red
                'duration': 'long',           # Continuous
                'objects': ['fire', 'smoke'],
                'static_location': True,      # Doesn't move much
                'upward': True                # Flames rise
            },
            'smoke': {
                'motion_type': 'slow_drift',
                'motion_range': (5, 50),      # Very slow
                'color_pattern': 'gray_black',
                'duration': 'long',           # Continuous
                'objects': ['smoke'],
                'diffused': True,             # Spreads slowly
                'low_contrast': True
            },
            'fighting': {
                'motion_type': 'irregular_chaotic',
                'motion_range': (60, 200),    # Erratic
                'color_pattern': 'normal',    # Natural scene
                'duration': 'medium',         # 2-5 seconds
                'objects': ['person', 'people'],
                'irregular': True,            # Back-and-forth
                'multiple_humans': True
            }
        }
    
    def differentiate_accident_vs_explosion(self, motion_spike, brightness, orange_colors, yolo_vehicles):
        """
        Key differentiation:
        - Accident: 100-300 motion spike, neutral colors, vehicles present
        - Explosion: 300+ motion spike, bright flash, orange colors
        """
        if motion_spike > 300 or brightness > 200 or orange_colors > 5.0:
            return 'explosion', f"Explosion: spike={motion_spike}, bright={brightness}, orange={orange_colors:.1f}%"
        elif 100 < motion_spike < 300 and yolo_vehicles > 20.0 and orange_colors < 3.0:
            return 'accident', f"Accident: spike={motion_spike}, vehicles={yolo_vehicles:.0f}%, no flash"
        else:
            return 'neither', f"Unclear: spike={motion_spike}, bright={brightness}"
    
    def differentiate_fire_vs_explosion(self, duration_frames, flicker_variance, static_location):
        """
        Key differentiation:
        - Fire: Long duration (>30 frames), flickering, static location
        - Explosion: Very short (<10 frames), sudden burst, expanding
        """
        if duration_frames > 30 and flicker_variance > 15 and static_location:
            return 'fire', f"Fire: duration={duration_frames} frames, flickering={flicker_variance:.1f}"
        elif duration_frames < 10:
            return 'explosion', f"Explosion: short burst ({duration_frames} frames)"
        else:
            return 'unclear', f"Duration={duration_frames} frames"
    
    def differentiate_fighting_vs_accident(self, human_count, motion_pattern, vehicle_present):
        """
        Key differentiation:
        - Fighting: Multiple humans (2+), irregular motion, no vehicles
        - Accident: Vehicles present, directional motion
        """
        if human_count >= 2 and motion_pattern == 'irregular' and not vehicle_present:
            return 'fighting', f"Fighting: {human_count} people, irregular motion"
        elif vehicle_present and motion_pattern == 'linear':
            return 'accident', f"Accident: vehicles present, linear collision"
        else:
            return 'unclear', f"humans={human_count}, vehicles={vehicle_present}"
    
    def analyze_motion_pattern(self, optical_flow):
        """
        Analyze optical flow to determine motion type:
        - Linear: Directional (accident)
        - Radial: Outward burst (explosion)
        - Upward: Rising flames (fire)
        - Irregular: Chaotic (fighting)
        - Slow: Drifting (smoke)
        """
        # This would use actual optical flow vectors
        # For now, placeholder logic
        pass
    
    def get_event_confidence(self, event_type, indicators):
        """
        Calculate confidence based on how many signature indicators match
        Returns confidence (0-100) and reasoning
        """
        signature = self.signatures.get(event_type, {})
        matches = 0
        total_checks = 0
        reasons = []
        
        # Check motion range
        if 'motion_spike' in indicators:
            total_checks += 1
            motion_range = signature.get('motion_range', (0, 0))
            if motion_range[0] <= indicators['motion_spike'] <= motion_range[1]:
                matches += 1
                reasons.append(f"Motion in range ({motion_range[0]}-{motion_range[1]})")
        
        # Check color pattern
        if 'color_match' in indicators:
            total_checks += 1
            if indicators['color_match']:
                matches += 1
                reasons.append(f"Color pattern matches {signature.get('color_pattern')}")
        
        # Check object detection
        if 'objects_detected' in indicators:
            total_checks += 1
            expected_objects = signature.get('objects', [])
            if any(obj in indicators['objects_detected'] for obj in expected_objects):
                matches += 1
                reasons.append(f"Expected objects detected: {expected_objects}")
        
        if total_checks == 0:
            return 0.0, "No indicators provided"
        
        confidence = (matches / total_checks) * 100
        reasoning = " | ".join(reasons) if reasons else "No matches"
        
        return confidence, reasoning
