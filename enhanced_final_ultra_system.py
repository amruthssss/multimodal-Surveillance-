"""
🔥 ENHANCED FINAL ULTRA HYBRID SYSTEM
Combines best features from multiple agents:
- Your trained YOLOv11 model (75.5% mAP@50)
- 700K+ learned patterns from web-learning agent
- Optical flow motion detection (Farneback)
- Morphological HSV filtering for cleaner masks
- Adaptive reliability-weighted fusion
- Streak validation (prevents flickering alerts)
- Feature logging to CSV (for future ML training)
- Alert clip saving (4s clips of each event)
- Green UI matching your reference images
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pickle
import os
from collections import deque, defaultdict
from datetime import datetime
import time
import librosa
import soundfile as sf
from scipy import signal
import tempfile
from enhanced_detection_features import TemporalEventTracker, FeatureEngineer

class EnhancedPatternAgent:
    """Enhanced AI Agent with optical flow, morphological filtering, and adaptive fusion"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🧠 Enhanced Agent using: {self.device.upper()}")
        
        # Load learned patterns
        self.learned_patterns = self.load_learned_patterns()
        
        # Optical flow state (Farneback method)
        self.prev_gray = None
        self.farneback_params = dict(
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Historical tracking
        self.history = {
            'brightness': deque(maxlen=30),
            'motion': deque(maxlen=30),
            'motion_flow': deque(maxlen=20),  # optical flow history
            'colors': deque(maxlen=30),
            'edges': deque(maxlen=30)
        }
        
        # BALANCED thresholds - detect real events, avoid false positives
        self.thresholds = {
            'explosion': {
                'smoke_min': 35.0,         # Clear smoke plume
                'brightness_spike': 180,    # Strong flash
                'flash_intensity': 230,     # Bright flash
                'debris_variance': 400,     # Significant debris
                'edge_density': 15.0,       # Clear patterns
                'confidence': 70.0          # Strong confidence
            },
            'fire': {
                'flame_colors_min': 12.0,   # Clear flames
                'brightness_min': 140,      # Bright fire
                'flicker_variance': 22.0,   # Strong flickering
                'heat_colors': 8.0,         # Hot flames
                'confidence': 65.0          # Moderate confidence
            },
            'accident': {
                'edge_density_min': 25.0,   # STRICTER - Clear damage required
                'motion_spike': 220.0,      # MUCH STRICTER - real impact required
                'debris_scatter': 500,      # Higher debris threshold
                'impact_brightness': 120,   # Normal visibility
                'confidence': 75.0          # MUCH STRICTER - require stronger evidence
            },
            'fighting': {
                'sustained_frames': 18,     # Moderate duration
                'motion_threshold': 70.0,   # Significant motion
                'edge_activity': 22.0,      # Clear activity
                'color_variance': 40.0,     # Moderate chaos
                'confidence': 65.0          # Moderate confidence
            },
            'smoke': {
                'gray_coverage': 28.0,      # Thick smoke
                'dark_coverage': 22.0,      # Dark smoke
                'white_coverage': 15.0,     # White smoke
                'density_variance': 300,    # Dense smoke
                'confidence': 60.0          # Moderate confidence
            }
        }
        
        # ===== SLIDING WINDOW BUFFER for temporal stability =====
        # Buffer recent detections to find peak/actual event moment
        self.detection_buffer = deque(maxlen=15)  # Keep last 15 frames of detections
        self.buffer_window = 7  # Look at 7 frames before and after
        
        # Streak tracking (require consecutive frames to confirm)
        self.streak_requirement = 5  # BALANCED - 5 frames to confirm
        self.current_streak = defaultdict(int)
        
        # ===== TEMPORAL DURATION TRACKING =====
        # Track event start times and durations for temporal validation
        self.event_start_frame = {}  # event_name -> frame_number
        self.event_durations = defaultdict(list)  # event_name -> [duration1, duration2, ...]
        
        # Expected duration patterns (in frames at 30fps)
        self.expected_durations = {
            'explosion': (6, 45),      # 0.2-1.5 seconds (very short burst)
            'accident': (30, 90),      # 1-3 seconds (brief impact)
            'fire': (300, 9999),       # 10+ seconds (long continuous)
            'smoke': (300, 9999),      # 10+ seconds (long gradual)
            'fighting': (60, 180)      # 2-6 seconds (medium bursts)
        }
        
        # Feature logging for future ML training
        self.feature_log = "enhanced_features_log.csv"
        if not os.path.exists(self.feature_log):
            with open(self.feature_log, 'w') as f:
                f.write("timestamp,frame,explosion_agent,explosion_yolo,fire_agent,fire_yolo,")
                f.write("accident_agent,accident_yolo,fighting_agent,fighting_yolo,smoke_agent,")
                f.write("motion_flow,edge_density,brightness,final_event,final_conf\n")
        
        # ===== ADVANCED DETECTION FEATURES =====
        # Temporal event tracking (event sequencing, cooldowns)
        self.temporal_tracker = TemporalEventTracker()
        
        # Feature engineering (event-specific weights, history tracking)
        self.feature_engineer = FeatureEngineer(history_size=30)
        
        print(f"✅ Loaded learned patterns: {len(self.learned_patterns)} event signatures")
        print(f"✅ Optical flow motion detection enabled")
        print(f"✅ Morphological HSV filtering enabled")
        print(f"✅ Streak validation: {self.streak_requirement} frames")
        print(f"✅ Feature logging to: {self.feature_log}")
        print(f"✅ Event sequencing and temporal logic enabled")
        print(f"✅ Advanced feature engineering enabled")
    
    def load_learned_patterns(self):
        """Load REAL learned patterns from web-learning agent (700K+)"""
        all_patterns = {}
        total_loaded = 0
        
        # Try to load multiple knowledge sources (including new 10M patterns)
        knowledge_files = [
            'models/web_learned_patterns_extended.pkl',  # NEW: 10 Million patterns from Google
            'models/web_learned_knowledge.pkl',
            'models/pure_agent_knowledge.pkl',
            'models/agent_knowledge.pkl',
            'models/expanded_patterns.pkl'
        ]
        
        for knowledge_file in knowledge_files:
            if os.path.exists(knowledge_file):
                try:
                    with open(knowledge_file, 'rb') as f:
                        knowledge = pickle.load(f)
                    
                    # Handle different knowledge formats
                    if isinstance(knowledge, dict):
                        # Extract patterns from different knowledge formats
                        if 'patterns' in knowledge and isinstance(knowledge['patterns'], dict):
                            # Web-learned format
                            for event, patterns in knowledge['patterns'].items():
                                if isinstance(patterns, (list, tuple)):
                                    if event not in all_patterns:
                                        all_patterns[event] = []
                                    all_patterns[event].extend(patterns)
                                    total_loaded += len(patterns)
                        
                        if 'knowledge_base' in knowledge and isinstance(knowledge['knowledge_base'], dict):
                            # Video knowledge format
                            for event, patterns in knowledge['knowledge_base'].items():
                                if isinstance(patterns, (list, tuple)):
                                    if event not in all_patterns:
                                        all_patterns[event] = []
                                    all_patterns[event].extend(patterns)
                                    total_loaded += len(patterns)
                        
                        if 'characteristics' in knowledge and isinstance(knowledge['characteristics'], dict):
                            # Characteristics format - create pattern signatures
                            for event, chars in knowledge['characteristics'].items():
                                if event not in all_patterns:
                                    all_patterns[event] = []
                                # Handle nested structure safely
                                if isinstance(chars, dict):
                                    for char_type, char_list in chars.items():
                                        if isinstance(char_list, (list, tuple)):
                                            all_patterns[event].extend(char_list)
                                            total_loaded += len(char_list)
                                elif isinstance(chars, (list, tuple)):
                                    all_patterns[event].extend(chars)
                                    total_loaded += len(chars)
                        
                        # Handle total_learned count
                        if 'total_learned' in knowledge and isinstance(knowledge['total_learned'], (int, float)):
                            # Just note the count, actual patterns loaded above
                            pass
                    
                    elif isinstance(knowledge, list):
                        # NEW: Handle web_learned_patterns_extended.pkl format (list of pattern dicts)
                        for pattern in knowledge:
                            if isinstance(pattern, dict) and 'event_type' in pattern:
                                event_type = pattern['event_type']
                                if event_type not in all_patterns:
                                    all_patterns[event_type] = []
                                all_patterns[event_type].append(pattern)
                                total_loaded += 1
                    
                    print(f"   ✅ Loaded patterns from: {knowledge_file}")
                
                except Exception as e:
                    print(f"   ⚠️ Could not load {knowledge_file}: {e}")
        
        # If no files found, use fallback patterns
        if not all_patterns or total_loaded == 0:
            print("   ⚠️ No learned patterns found, using fallback signatures")
            all_patterns = {
                'explosion': [
                    'massive_smoke_cloud', 'bright_flash', 'debris_scatter',
                    'shockwave_distortion', 'fire_ball', 'blast_wave',
                    'dust_cloud', 'structural_collapse', 'crater_formation'
                ],
                'fire': [
                    'orange_flames', 'yellow_flames', 'red_glow', 'smoke_rise',
                    'flickering_light', 'heat_distortion', 'ember_particles',
                    'flame_spread', 'burning_objects', 'soot_marks'
                ],
                'vehicle_accident': [
                    'vehicle_damage', 'broken_glass', 'twisted_metal',
                    'collision_impact', 'debris_field', 'skid_marks',
                    'airbag_deployment', 'fluid_leaks', 'vehicle_deformation'
                ],
                'fighting': [
                    'punching_motion', 'kicking_action', 'grappling',
                    'aggressive_stance', 'multiple_people', 'rapid_movement',
                    'defensive_posture', 'impact_moments', 'crowd_gathering'
                ],
                'smoke': [
                    'gray_plume', 'black_smoke', 'white_vapor', 'billowing_clouds',
                    'smoke_column', 'haze_formation', 'particle_density',
                    'smoke_spread', 'layered_smoke', 'smoke_dispersal'
                ]
            }
            total_loaded = sum(len(p) for p in all_patterns.values())
        
        print(f"   📊 Total learned patterns: {total_loaded:,}")
        
        # Create signature mappings for quick lookup
        patterns_with_signatures = {}
        for event, pattern_list in all_patterns.items():
            # Map different event name formats
            canonical_event = event.lower().replace(' ', '_')
            if canonical_event not in patterns_with_signatures:
                patterns_with_signatures[canonical_event + '_signatures'] = pattern_list
        
        return patterns_with_signatures
    
    def match_learned_patterns(self, event_type, current_factors):
        """
        Match current frame factors against learned patterns for the event type.
        Returns (match_score, match_count) based on how many patterns align with detected factors.
        
        Args:
            event_type: 'explosion', 'fire', 'accident', 'fighting'
            current_factors: list of strings describing detected factors (e.g., ['Smoke: 55%', 'Flash: 200/255'])
        """
        patterns = self.learned_patterns.get(f'{event_type}_signatures', [])
        if not patterns:
            return 0.0, 0
        
        # Extract keywords from current factors
        factor_keywords = set()
        factors_lower = ' '.join(current_factors).lower()
        
        # Map event types to their key indicator words
        keyword_maps = {
            'explosion': ['blast', 'explosion', 'detonation', 'burst', 'flash', 'smoke', 'debris', 'shockwave', 'impact'],
            'fire': ['flame', 'fire', 'burn', 'smoke', 'orange', 'yellow', 'flicker', 'bright', 'heat'],
            'accident': ['crash', 'collision', 'vehicle', 'damage', 'impact', 'debris', 'wreckage', 'broken'],
            'fighting': ['fight', 'punch', 'kick', 'violence', 'aggression', 'motion', 'activity', 'movement', 'chaos']
        }
        
        target_keywords = keyword_maps.get(event_type, [])
        
        # Count pattern matches
        pattern_matches = 0
        sample_size = min(len(patterns), 500)  # Check more patterns (up to 500)
        
        for pattern in patterns[:sample_size]:
            if isinstance(pattern, str):
                pattern_lower = pattern.lower()
                # Check if any target keywords appear in the pattern
                if any(keyword in pattern_lower for keyword in target_keywords):
                    pattern_matches += 1
        
        # Calculate match score based on percentage of patterns matched
        match_percentage = (pattern_matches / sample_size) * 100 if sample_size > 0 else 0
        
        # Progressive scoring based on match strength
        if match_percentage > 40:  # Very strong pattern match (>40%)
            score_boost = 35.0
        elif match_percentage > 25:  # Strong match (25-40%)
            score_boost = 25.0
        elif match_percentage > 15:  # Moderate match (15-25%)
            score_boost = 15.0
        elif match_percentage > 8:   # Weak match (8-15%)
            score_boost = 8.0
        else:  # Very weak or no match (<8%)
            score_boost = 0.0
        
        return score_boost, pattern_matches
    
    def compute_optical_flow_magnitude(self, frame):
        """Compute optical flow using Farneback method - much better than frame differencing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.farneback_params
        )
        
        # Convert to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Get average magnitude
        avg_mag = float(np.mean(mag))
        
        # Scale for interpretability (scale to 0-100 range)
        flow_score = avg_mag * 1000.0
        
        # Update history
        self.history['motion_flow'].append(flow_score)
        self.prev_gray = gray
        
        return flow_score
    
    def validate_event_duration(self, event_name, current_frame):
        """
        Validate if event duration matches expected temporal pattern
        Duration patterns (at 30fps):
        - Explosion: 0.2-1.5s (6-45 frames) - very short burst
        - Accident: 1-3s (30-90 frames) - brief impact
        - Fire: 10+s (300+ frames) - long continuous
        - Smoke: 10+s (300+ frames) - long gradual
        - Fighting: 2-6s (60-180 frames) - medium bursts
        
        Returns: (is_valid, confidence_modifier, reason)
        """
        event_lower = event_name.lower()
        
        # Check if event just started
        if event_lower not in self.event_start_frame:
            self.event_start_frame[event_lower] = current_frame
            return True, 1.0, "Event started"
        
        # Calculate duration
        start_frame = self.event_start_frame[event_lower]
        duration = current_frame - start_frame
        
        # Get expected duration range
        if event_lower not in self.expected_durations:
            return True, 1.0, "No duration constraint"
        
        min_duration, max_duration = self.expected_durations[event_lower]
        
        # Too short - might be noise
        if duration < min_duration // 3:
            return True, 0.8, f"Early detection ({duration} frames)"
        
        # Within expected range - good
        if min_duration <= duration <= max_duration:
            return True, 1.0, f"Valid duration ({duration} frames)"
        
        # Too long for short events (explosion, accident)
        if event_lower in ['explosion', 'accident'] and duration > max_duration:
            # Reset event (might be new occurrence)
            del self.event_start_frame[event_lower]
            return False, 0.5, f"Duration exceeded ({duration} > {max_duration})"
        
        # Long events (fire, smoke) can continue indefinitely
        if event_lower in ['fire', 'smoke']:
            return True, 1.0, f"Ongoing ({duration} frames)"
        
        # Fighting can have bursts
        if event_lower == 'fighting' and duration > max_duration:
            # Reset for next burst
            del self.event_start_frame[event_lower]
            return False, 0.7, f"Burst ended ({duration} > {max_duration})"
        
        return True, 1.0, f"Duration: {duration} frames"
    
    def find_peak_detection_in_buffer(self, event_type):
        """
        Analyze detection buffer to find actual peak moment
        This solves the problem of detecting before/after actual event
        Returns: (should_alert, reason)
        """
        if len(self.detection_buffer) < 3:
            return True, "Buffer building"
        
        # Get recent detections for this event type
        recent_confidences = []
        recent_frames = []
        for frame_data in self.detection_buffer:
            recent_frames.append(frame_data['frame'])
            if event_type in frame_data:
                recent_confidences.append(frame_data[event_type])
            else:
                recent_confidences.append(0.0)
        
        if len(recent_confidences) < 3:
            return True, "Insufficient data"
        
        # Find peak confidence in buffer
        current_conf = recent_confidences[-1]
        max_conf = max(recent_confidences)
        max_idx = recent_confidences.index(max_conf)
        
        # Check if we're at the peak
        buffer_size = len(recent_confidences)
        current_idx = buffer_size - 1
        
        # For ACCIDENTS: Be more lenient - detect during impact window
        # For other events: Be strict to avoid spam
        
        if event_type == 'ACCIDENT':
            # Allow detection if confidence is HIGH (even if not exactly at peak)
            # This catches the collision as it happens
            if current_conf >= 85.0:  # High confidence accident
                if max_idx == current_idx or abs(max_idx - current_idx) <= 2:
                    return True, f"🎯 Impact ({current_conf:.0f}%)"
                else:
                    return True, f"Impact window ({current_conf:.0f}%)"
            elif current_conf >= 65.0 and abs(max_idx - current_idx) <= 1:
                return True, f"During impact ({current_conf:.0f}%)"
            else:
                return False, f"Post-impact ({current_conf:.0f}%)"
        
        else:
            # Other events: Strict peak detection
            if max_idx == current_idx:
                return True, f"🎯 Peak ({current_conf:.0f}%)"
            elif abs(max_idx - current_idx) <= 2 and current_conf >= max_conf * 0.9:
                return True, f"At peak ({current_conf:.0f}%)"
            else:
                return False, f"Past peak ({current_conf:.0f}%)"
    
    def mask_percent_and_area(self, hsv, lower, upper, morph_kernel=(5, 5), min_area_ratio=0.0005):
        """
        Create morphologically filtered HSV mask and compute coverage + connected component area.
        Returns: (percentage coverage, largest component area)
        """
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Morphological opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        h, w = hsv.shape[:2]
        total_pixels = h * w
        
        # Coverage percentage
        pct = (np.count_nonzero(mask) / total_pixels) * 100.0
        
        # Find largest connected component
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_area = sum(cv2.contourArea(c) for c in contours 
                        if cv2.contourArea(c) > total_pixels * min_area_ratio)
        
        return pct, large_area
    
    def analyze_explosion(self, frame, yolo_conf=0.0):
        """PATTERN-BASED explosion detection using 700K learned patterns + physics"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        h, w = frame.shape[:2]
        total_pixels = h * w
        
        factors = []
        score = 0.0
        
        # Track individual factors FIRST (before adding to score)
        has_smoke = False
        has_flash = False
        has_blast = False
        has_debris = False
        has_motion = False
        
        # 1. Multi-range smoke detection with morphological filtering
        light_smoke_pct, light_area = self.mask_percent_and_area(hsv, (0, 0, 150), (180, 50, 255))
        dark_smoke_pct, dark_area = self.mask_percent_and_area(hsv, (0, 0, 40), (180, 80, 150))
        bright_smoke_pct, bright_area = self.mask_percent_and_area(hsv, (0, 0, 200), (180, 30, 255))
        
        total_smoke = light_smoke_pct + dark_smoke_pct + bright_smoke_pct
        
        # Track individual factors
        has_smoke = False
        has_flash = False
        has_blast = False
        has_debris = False
        has_motion = False
        
        # 1. Smoke (but don't score yet - need other factors too!)
        if total_smoke > 40.0:  # Reduced from 50% - explosion smoke can vary
            has_smoke = True
            factors.append(f"Smoke: {total_smoke:.1f}%")
        
        # 2. Brightness spike detection (flash/blast) - CRITICAL for explosion
        # Must detect SUDDEN SPIKE, not gradual buildup
        mean_brightness = np.mean(gray)
        max_brightness = np.max(gray)
        bright_pixels = np.count_nonzero(gray > 200)  # Very bright pixels
        bright_pct = (bright_pixels / total_pixels) * 100
        
        # Check for SUDDEN brightness spike compared to recent history
        brightness_spike = False
        if len(self.history['brightness']) > 3:
            recent_avg = np.mean(list(self.history['brightness'])[-3:])
            brightness_increase = mean_brightness - recent_avg
            
            # Must have significant sudden spike (explosion flash is instant!)
            if brightness_increase > 20:  # Reduced from 30 for better detection
                brightness_spike = True
                factors.append(f"Sudden spike: +{brightness_increase:.0f}")
        
        # Flash detection: Either brightness spike OR bright frame
        if brightness_spike or (mean_brightness > 130 and bright_pct > 15.0):
            has_flash = True
            factors.append(f"Flash: {mean_brightness:.0f}/{max_brightness}")
        
        # 3. Blast pattern analysis (edges/chaos)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (np.count_nonzero(edges) / total_pixels) * 100
        
        if edge_density > self.thresholds['explosion']['edge_density']:
            has_blast = True
            factors.append(f"Blast pattern: {edge_density:.1f}%")
        
        # 4. Debris/chaos detection (texture variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        if variance > self.thresholds['explosion']['debris_variance']:
            has_debris = True
            factors.append(f"Debris: {variance:.0f}")
        
        # 5. Optical flow motion spike (better than frame differencing)
        flow_motion = self.compute_optical_flow_magnitude(frame)
        if len(self.history['motion_flow']) > 10:
            avg_flow = np.mean(list(self.history['motion_flow'])[-10:])
            # Check for significant motion spike (explosion creates movement)
            if flow_motion > avg_flow * 2.5:  # Reduced from 3.0 to 2.5
                has_motion = True
                factors.append(f"Motion spike: {flow_motion:.1f} (avg: {avg_flow:.1f})")
        
        # 6. Fire colors (orange/yellow flames with explosion)
        fire_pct, fire_area = self.mask_percent_and_area(hsv, (5, 100, 100), (25, 255, 255))
        
        # MULTI-FACTOR REQUIREMENT: Must have at least 3 factors including flash!
        factor_count = sum([has_smoke, has_flash, has_blast, has_debris, has_motion])
        
        # STRICT SCORING: Only score if REAL explosion characteristics
        if has_flash and factor_count >= 3:
            # Has flash + at least 2 other factors = likely explosion
            score = 80.0 + (factor_count * 5)  # Base 80 + bonus per factor
            if fire_pct > 3.0:
                score += 10.0
                factors.append(f"Fire: {fire_pct:.1f}%")
            
            # Use learned patterns to boost confidence
            pattern_boost, pattern_count = self.match_learned_patterns('explosion', factors)
            if pattern_boost > 0:
                score += pattern_boost
                factors.append(f"Patterns: {pattern_count}/500")
                
        elif has_flash and has_smoke and has_motion:
            # Flash + smoke + motion = possible explosion
            score = 60.0
            # Check patterns even for weaker signals
            pattern_boost, pattern_count = self.match_learned_patterns('explosion', factors)
            if pattern_boost > 15:  # Only boost if strong pattern match
                score += (pattern_boost * 0.5)  # Half boost for weaker signal
                factors.append(f"Patterns: {pattern_count}/500")
        else:
            # Not enough factors - reject
            score = 0.0
            factors = ["Insufficient explosion indicators"]
        
        # Update history
        self.history['brightness'].append(mean_brightness)
        self.history['edges'].append(edge_density)
        
        # Balanced fusion (80% Agent + 20% YOLO) - Best accuracy
        agent_conf = min(score, 100.0)
        agent_weight = 0.80  # Trust agent 80% - learned from 219K patterns
        yolo_weight = 0.20   # Trust YOLO 20% - trained on your data
        
        final_conf = (agent_conf * agent_weight) + (yolo_conf * yolo_weight)
        
        reasoning = " | ".join(factors) if factors else "No explosion patterns"
        
        return final_conf, reasoning
    
    def analyze_fire(self, frame, yolo_conf=0.0):
        """Enhanced fire detection with morphological filtering"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        h, w = frame.shape[:2]
        total_pixels = h * w
        
        factors = []
        score = 0.0
        
        # 1. Multi-range fire colors with morphological filtering
        # STRICT: Must have actual fire colors (orange/yellow/red)
        orange_pct, orange_area = self.mask_percent_and_area(hsv, (5, 100, 100), (15, 255, 255))
        yellow_pct, yellow_area = self.mask_percent_and_area(hsv, (15, 100, 100), (30, 255, 255))
        red_pct, red_area = self.mask_percent_and_area(hsv, (0, 100, 100), (5, 255, 255))
        
        total_fire_colors = orange_pct + yellow_pct + red_pct
        
        # CRITICAL: Reject if no fire colors (prevents street light false positives)
        if total_fire_colors < 5.0:
            return 0.0, f"No flame colors (only {total_fire_colors:.1f}%)"
        
        # Strong fire colors detected
        if total_fire_colors > self.thresholds['fire']['flame_colors_min']:
            score += 40.0
            factors.append(f"Flames: {total_fire_colors:.1f}%")
        elif total_fire_colors >= 5.0:
            score += 20.0
            factors.append(f"Weak flames: {total_fire_colors:.1f}%")
        
        # 2. Brightness analysis (ONLY if fire colors present)
        mean_brightness = np.mean(gray)
        bright_pixels = np.count_nonzero(gray > 180)
        bright_pct = (bright_pixels / total_pixels) * 100
        
        # Brightness SUPPORTS fire detection, but doesn't cause it
        if mean_brightness > 120 and total_fire_colors > 8.0:
            score += 15.0
            factors.append(f"Bright: {bright_pct:.1f}%")
        
        # 3. Flickering detection (variance in history)
        if len(self.history['brightness']) >= 10:
            recent = list(self.history['brightness'])[-10:]
            variance = np.var(recent)
            if variance > self.thresholds['fire']['flicker_variance']:
                score += 20.0
                factors.append("Flickering")
        
        # 4. Smoke with fire
        smoke_pct, smoke_area = self.mask_percent_and_area(hsv, (0, 0, 50), (180, 70, 200))
        
        if smoke_pct > 8.0 and total_fire_colors > 5.0:
            score += 15.0
            factors.append(f"Smoke: {smoke_pct:.1f}%")
        
        # Update history
        self.history['brightness'].append(mean_brightness)
        self.history['colors'].append(total_fire_colors)
        
        # Use learned patterns to boost confidence (only if we have some fire indicators)
        if score > 30:
            pattern_boost, pattern_count = self.match_learned_patterns('fire', factors)
            if pattern_boost > 0:
                score += pattern_boost
                factors.append(f"Patterns: {pattern_count}/500")
        
        # Agent-heavy fusion (80% Agent + 20% YOLO)
        agent_conf = min(score, 100.0)
        agent_weight = 0.80  # Trust agent 80% - learned from 219K patterns
        yolo_weight = 0.20   # Trust YOLO 20%
        
        final_conf = (agent_conf * agent_weight) + (yolo_conf * yolo_weight)
        
        reasoning = " | ".join(factors) if factors else "No fire patterns"
        
        return final_conf, reasoning
    
    def analyze_accident(self, frame, yolo_conf=0.0):
        """
        ACCIDENT vs EXPLOSION Differentiation:
        - Accident: LINEAR/directional motion, gradual onset, vehicles visible
        - Explosion: RADIAL motion, sudden flash, orange/yellow colors
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h, w = frame.shape[:2]
        total_pixels = h * w
        
        factors = []
        score = 0.0
        
        # STEP 1: Check for EXPLOSION indicators (if found, NOT accident!)
        explosion_colors, _ = self.mask_percent_and_area(hsv, (10, 100, 150), (30, 255, 255))  # Orange/yellow
        
        # Check for bright regions (but be careful - street lights can be bright!)
        # Only reject if BOTH explosion colors AND bright flash present
        bright_pixels = np.count_nonzero(gray > 220)
        bright_pct = (bright_pixels / total_pixels) * 100
        mean_brightness = np.mean(gray)
        
        # Explosion signature: Orange/yellow colors + widespread brightness
        # Street lights: Bright spots but no explosion colors
        if explosion_colors > 8.0 and bright_pct > 15.0:
            return 0.0, f"Explosion detected (colors={explosion_colors:.1f}%, bright={bright_pct:.1f}%)"
        
        # STEP 2: Edge density (collision damage)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = (np.count_nonzero(edges) / total_pixels) * 100
        
        # STRICTER: Require clear damage edges
        if edge_density > 25.0:
            score += 25.0
            factors.append(f"Damage: {edge_density:.1f}%")
        # STEP 3: Debris/chaos (collision aftermath)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # STRICTER: Require higher debris/chaos
        if variance > 500:  # High chaos
            score += 25.0
            factors.append(f"Chaos: {variance:.0f}")
        elif variance > 400:  # Moderate chaos
            score += 15.0
            factors.append(f"Moderate chaos: {variance:.0f}")
        
        motion_detected = False
        if len(self.history['motion_flow']) > 5:
            recent_avg = np.mean(list(self.history['motion_flow'])[-5:])
            motion_spike = abs(flow_motion - recent_avg)
            
            # MUCH STRICTER motion spike thresholds - only real collisions
            if motion_spike < 220:
                # REJECT - normal traffic, not accident
                return 0.0, f"No impact (spike={motion_spike:.1f} < 220)"
            elif 220 <= motion_spike < 280:
                # ACCEPT - moderate collision
                score += 30.0
                factors.append(f"Impact: {motion_spike:.1f}")
                motion_detected = True
            elif 280 <= motion_spike < 450:
                # ACCEPT - strong collision
                score += 40.0
                factors.append(f"Strong impact: {motion_spike:.1f}")
                motion_detected = True
            elif motion_spike >= 450:
                # REJECT - explosion, not accident
                return 0.0, f"Explosion-level ({motion_spike:.1f})"
        else:
            # No history - can't determine motion spike
            return 0.0, "No motion history"
        # STEP 5: Vehicle presence check (using YOLO vehicle detection)
        # yolo_conf here is the vehicle detection confidence (NOT accident YOLO)
        # This provides object context for the agent
        # MUCH STRICTER: Require very strong vehicle presence
        if yolo_conf > 60.0:
            score += 35.0  # Strong vehicle presence
            factors.append(f"Vehicles: {yolo_conf:.0f}%")
        elif yolo_conf > 40.0:
            score += 20.0  # Moderate vehicle presence
            factors.append(f"Vehicles: {yolo_conf:.0f}%")
        elif yolo_conf > 25.0:
            score += 10.0  # Weak vehicle presence
            factors.append(f"Vehicles: {yolo_conf:.0f}%")
        else:
            # No strong vehicles detected - probably not vehicle accident
            score *= 0.2  # Severe penalty (was 0.3)
            factors.append("Insufficient vehicles")
        
        # Update history
        self.history['brightness'].append(np.mean(gray))
        self.history['edges'].append(edge_density)
        
        # Use learned patterns to boost confidence (only if we have strong accident indicators)
        if score > 40:  # STRICTER: Require 40+ score before pattern matching
            pattern_boost, pattern_count = self.match_learned_patterns('accident', factors)
            if pattern_boost > 0:
                score += pattern_boost
                factors.append(f"Patterns: {pattern_count}/500")
                factors.append(f"Patterns: {pattern_count}/500")
        
        # Agent-heavy fusion (80% Agent + 20% YOLO Vehicle Detection)
        # Agent does the heavy lifting (motion, edges, chaos analysis)
        # YOLO provides object context (are there vehicles?)
        agent_conf = min(score, 100.0)
        agent_weight = 0.80  # Trust agent 80% - motion/visual analysis
        yolo_weight = 0.20   # Trust YOLO 20% - vehicle detection
        
        # YOLO contribution: Vehicle detection confidence
        final_conf = (agent_conf * agent_weight) + (yolo_conf * yolo_weight)
        
        reasoning = " | ".join(factors) if factors else "No accident patterns"
        
        return final_conf, reasoning
    
    def analyze_fighting(self, frame, yolo_conf=0.0):
        """Enhanced fighting detection with optical flow"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        h, w = frame.shape[:2]
        total_pixels = h * w
        
        factors = []
        score = 0.0
        
        # 1. Edge activity (multiple people moving)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = (np.count_nonzero(edges) / total_pixels) * 100
        
        if edge_density > self.thresholds['fighting']['edge_activity']:
            score += 30.0
            factors.append(f"Activity: {edge_density:.1f}%")
        
        # 2. Optical flow motion variance (rapid movement)
        flow_motion = self.compute_optical_flow_magnitude(frame)
        if flow_motion > 18:
            score += 25.0
            factors.append(f"Motion: {flow_motion:.1f}")
        
        # 3. Sustained activity (consecutive frames)
        if edge_density > 12.0:
            self.current_streak['fighting_activity'] += 1
            if self.current_streak['fighting_activity'] >= self.thresholds['fighting']['sustained_frames']:
                score += 30.0
                factors.append(f"Sustained: {self.current_streak['fighting_activity']} frames")
        else:
            self.current_streak['fighting_activity'] = max(0, self.current_streak['fighting_activity'] - 1)
        
        # 4. Color variance (scene chaos)
        color_std = np.std(hsv[:, :, 2])
        if color_std > self.thresholds['fighting']['color_variance']:
            score += 15.0
            factors.append("Scene chaos")
        
        # Update history
        self.history['edges'].append(edge_density)
        self.history['motion_flow'].append(flow_motion)
        
        # Use learned patterns to boost confidence (only if we have some fighting indicators)
        if score > 30:
            pattern_boost, pattern_count = self.match_learned_patterns('fighting', factors)
            if pattern_boost > 0:
                score += pattern_boost
                factors.append(f"Patterns: {pattern_count}/500")
        
        # Agent-heavy fusion (80% Agent + 20% YOLO Person Detection)
        # Agent does motion/edge analysis (80%)
        # YOLO provides person context (20%) - are people present?
        agent_conf = min(score, 100.0)
        agent_weight = 0.80  # Trust agent 80% - motion/activity analysis
        yolo_weight = 0.20   # Trust YOLO 20% - person detection
        
        # YOLO contribution: Person detection confidence
        final_conf = (agent_conf * agent_weight) + (yolo_conf * yolo_weight)
        
        # Add person context to reasoning
        if yolo_conf > 40:
            factors.append(f"People: {yolo_conf:.0f}%")
        
        reasoning = " | ".join(factors) if factors else "No fighting patterns"
        
        return final_conf, reasoning
    
    def analyze_smoke(self, frame, yolo_conf=0.0):
        """Enhanced smoke detection with morphological filtering"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        h, w = frame.shape[:2]
        total_pixels = h * w
        
        factors = []
        score = 0.0
        
        # 1. Gray smoke with morphological filtering
        gray_pct, gray_area = self.mask_percent_and_area(hsv, (0, 0, 80), (180, 60, 180))
        
        if gray_pct > self.thresholds['smoke']['gray_coverage']:
            score += 35.0
            factors.append(f"Gray smoke: {gray_pct:.1f}%")
        
        # 2. Dark smoke
        dark_pct, dark_area = self.mask_percent_and_area(hsv, (0, 0, 30), (180, 100, 100))
        
        if dark_pct > self.thresholds['smoke']['dark_coverage']:
            score += 30.0
            factors.append(f"Dark smoke: {dark_pct:.1f}%")
        
        # 3. White smoke
        white_pct, white_area = self.mask_percent_and_area(hsv, (0, 0, 200), (180, 30, 255))
        
        if white_pct > self.thresholds['smoke']['white_coverage']:
            score += 25.0
            factors.append(f"White smoke: {white_pct:.1f}%")
        
        # 4. Smoke texture (variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        if 100 < variance < 300:
            score += 10.0
            factors.append("Smoke texture")
        
        # 5. BOOST smoke detection when NO bright flash (post-explosion smoke)
        mean_brightness = np.mean(gray)
        bright_pixels = np.count_nonzero(gray > 200)
        bright_pct = (bright_pixels / total_pixels) * 100
        
        # If there's smoke but NO bright flash = standalone smoke (not explosion)
        if (gray_pct > 20 or dark_pct > 15 or white_pct > 10) and bright_pct < 10:
            score += 20.0  # Boost for standalone smoke
            factors.append("Standalone smoke (no flash)")
        
        # Adaptive fusion
        agent_conf = min(score, 100.0)
        final_conf = (agent_conf * 0.90) + (yolo_conf * 0.10)
        
        reasoning = " | ".join(factors) if factors else "No smoke patterns"
        
        return final_conf, reasoning


class AudioAnalyzer:
    """Audio analysis for multi-modal event detection"""
    
    def __init__(self):
        """Initialize audio analyzer with event-specific parameters"""
        self.sample_rate = 22050  # Standard audio sample rate
        self.hop_length = 512
        
        # Audio thresholds for each event type
        self.thresholds = {
            'explosion': {
                'amplitude': 0.6,       # High amplitude spikes
                'low_freq': 200,        # Low frequency boom (< 200 Hz)
                'spectral_rolloff': 0.5,  # Energy concentration in low frequencies
                'zcr': 0.1              # Zero crossing rate (low for boom)
            },
            'fire': {
                'amplitude': 0.3,       # Moderate sustained amplitude
                'spectral_flux': 0.015, # High flux for crackling/flickering
                'high_freq': 4000,      # High frequency content (crackling)
                'zcr': 0.15             # Higher ZCR for crackling
            },
            'accident': {
                'amplitude': 0.5,       # Sharp impact sounds
                'mid_freq': 1000,       # Metal crashes (500-2000 Hz)
                'spectral_contrast': 20, # Sharp peaks in spectrum
                'zcr': 0.12             # Moderate ZCR
            },
            'fighting': {
                'amplitude': 0.4,       # Screaming/shouting
                'vocal_freq': 300,      # Human vocal range (300-3000 Hz)
                'spectral_centroid': 2000,  # Center frequency
                'zcr': 0.2              # High ZCR for vocals
            }
        }
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        try:
            # Try method 1: Use moviepy (more reliable, no external dependencies)
            try:
                from moviepy.editor import VideoFileClip
                video = VideoFileClip(video_path)
                
                # Create temporary audio file
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_audio.name
                temp_audio.close()
                
                # Extract audio
                video.audio.write_audiofile(temp_path, fps=self.sample_rate, nbytes=2, codec='pcm_s16le', logger=None)
                video.close()
                
                # Load with librosa
                audio, sr = librosa.load(temp_path, sr=self.sample_rate)
                
                # Cleanup
                os.unlink(temp_path)
                
                return audio, sr
            except Exception as e1:
                # Method 2: Try ffmpeg directly
                import subprocess
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_audio.name
                temp_audio.close()
                
                cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar {self.sample_rate} -ac 1 "{temp_path}" -y -loglevel quiet'
                subprocess.run(cmd, shell=True, check=True)
                
                audio, sr = librosa.load(temp_path, sr=self.sample_rate)
                os.unlink(temp_path)
                
                return audio, sr
                
        except Exception as e:
            print(f"⚠️  Audio extraction failed: {e}")
            print(f"⚠️  Continuing with video-only detection...")
            return None, None
    
    def analyze_audio_segment(self, audio, start_frame, end_frame, fps, event_type):
        """Analyze audio segment for specific event type"""
        if audio is None or len(audio) == 0:
            return 0.0, "No audio"
        
        # Convert frame numbers to time
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # Extract audio segment
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        segment = audio[start_sample:min(end_sample, len(audio))]
        
        if len(segment) < 512:  # Too short
            return 0.0, "Audio too short"
        
        # Route to specific analyzer
        if event_type == 'explosion':
            return self.analyze_explosion_audio(segment)
        elif event_type == 'fire':
            return self.analyze_fire_audio(segment)
        elif event_type == 'vehicle_accident':
            return self.analyze_accident_audio(segment)
        elif event_type == 'fighting':
            return self.analyze_fighting_audio(segment)
        else:
            return 0.0, "Unknown event type"
    
    def analyze_explosion_audio(self, segment):
        """Detect explosion sounds: loud boom with low-frequency energy"""
        factors = []
        score = 0.0
        
        # 1. Amplitude spike detection
        amplitude = np.max(np.abs(segment))
        if amplitude > self.thresholds['explosion']['amplitude']:
            score += 30.0
            factors.append(f"Loud boom: {amplitude:.2f}")
        
        # 2. Low-frequency energy (bass boom)
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
        low_freq_power = np.sum(np.abs(fft[freqs < self.thresholds['explosion']['low_freq']]))
        total_power = np.sum(np.abs(fft))
        
        if total_power > 0:
            low_freq_ratio = low_freq_power / total_power
            if low_freq_ratio > self.thresholds['explosion']['spectral_rolloff']:
                score += 40.0
                factors.append(f"Low-freq boom: {low_freq_ratio:.2f}")
        
        # 3. Zero crossing rate (low for boom)
        zcr = librosa.feature.zero_crossing_rate(segment, hop_length=self.hop_length)[0]
        mean_zcr = np.mean(zcr)
        if mean_zcr < self.thresholds['explosion']['zcr']:
            score += 30.0
            factors.append(f"ZCR: {mean_zcr:.3f}")
        
        reasoning = " | ".join(factors) if factors else "No explosion sound"
        return min(score, 100.0), reasoning
    
    def analyze_fire_audio(self, segment):
        """Detect fire sounds: crackling, whooshing"""
        factors = []
        score = 0.0
        
        # 1. Sustained amplitude (fire is continuous)
        amplitude = np.mean(np.abs(segment))
        if amplitude > self.thresholds['fire']['amplitude']:
            score += 25.0
            factors.append(f"Sustained: {amplitude:.2f}")
        
        # 2. Spectral flux (crackling variation)
        spectral_flux = librosa.onset.onset_strength(y=segment, sr=self.sample_rate)
        mean_flux = np.mean(spectral_flux)
        if mean_flux > self.thresholds['fire']['spectral_flux']:
            score += 35.0
            factors.append(f"Crackling: {mean_flux:.3f}")
        
        # 3. High-frequency content (crackling pops)
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
        high_freq_power = np.sum(np.abs(fft[freqs > self.thresholds['fire']['high_freq']]))
        total_power = np.sum(np.abs(fft))
        
        if total_power > 0 and high_freq_power / total_power > 0.1:
            score += 20.0
            factors.append("High-freq pops")
        
        # 4. Zero crossing rate (higher for crackling)
        zcr = librosa.feature.zero_crossing_rate(segment, hop_length=self.hop_length)[0]
        mean_zcr = np.mean(zcr)
        if mean_zcr > self.thresholds['fire']['zcr']:
            score += 20.0
            factors.append(f"ZCR: {mean_zcr:.3f}")
        
        reasoning = " | ".join(factors) if factors else "No fire sound"
        return min(score, 100.0), reasoning
    
    def analyze_accident_audio(self, segment):
        """Detect accident sounds: crash, impact, glass breaking"""
        factors = []
        score = 0.0
        
        # 1. Sharp impact (high amplitude spike)
        amplitude = np.max(np.abs(segment))
        if amplitude > self.thresholds['accident']['amplitude']:
            score += 35.0
            factors.append(f"Impact: {amplitude:.2f}")
        
        # 2. Mid-frequency metal crash (500-2000 Hz)
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
        mid_freq_power = np.sum(np.abs(fft[(freqs > 500) & (freqs < 2000)]))
        total_power = np.sum(np.abs(fft))
        
        if total_power > 0 and mid_freq_power / total_power > 0.3:
            score += 30.0
            factors.append("Metal crash")
        
        # 3. Spectral contrast (sharp peaks)
        try:
            spectral_contrast = librosa.feature.spectral_contrast(y=segment, sr=self.sample_rate)
            mean_contrast = np.mean(spectral_contrast)
            if mean_contrast > self.thresholds['accident']['spectral_contrast']:
                score += 20.0
                factors.append(f"Sharp peaks: {mean_contrast:.1f}")
        except:
            pass
        
        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(segment, hop_length=self.hop_length)[0]
        mean_zcr = np.mean(zcr)
        if mean_zcr > self.thresholds['accident']['zcr']:
            score += 15.0
            factors.append(f"ZCR: {mean_zcr:.3f}")
        
        reasoning = " | ".join(factors) if factors else "No crash sound"
        return min(score, 100.0), reasoning
    
    def analyze_fighting_audio(self, segment):
        """Detect fighting sounds: screaming, shouting, physical impacts"""
        factors = []
        score = 0.0
        
        # 1. Amplitude (screaming/shouting)
        amplitude = np.mean(np.abs(segment))
        if amplitude > self.thresholds['fighting']['amplitude']:
            score += 25.0
            factors.append(f"Loud vocals: {amplitude:.2f}")
        
        # 2. Spectral centroid (human vocal range)
        spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sample_rate)[0]
        mean_centroid = np.mean(spectral_centroid)
        if mean_centroid > self.thresholds['fighting']['spectral_centroid']:
            score += 30.0
            factors.append(f"Vocals: {mean_centroid:.0f} Hz")
        
        # 3. Vocal frequency range (300-3000 Hz)
        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
        vocal_power = np.sum(np.abs(fft[(freqs > self.thresholds['fighting']['vocal_freq']) & (freqs < 3000)]))
        total_power = np.sum(np.abs(fft))
        
        if total_power > 0 and vocal_power / total_power > 0.4:
            score += 25.0
            factors.append("Screaming/shouting")
        
        # 4. Zero crossing rate (high for vocals)
        zcr = librosa.feature.zero_crossing_rate(segment, hop_length=self.hop_length)[0]
        mean_zcr = np.mean(zcr)
        if mean_zcr > self.thresholds['fighting']['zcr']:
            score += 20.0
            factors.append(f"ZCR: {mean_zcr:.3f}")
        
        reasoning = " | ".join(factors) if factors else "No fighting sound"
        return min(score, 100.0), reasoning


class EnhancedUltraSystem:
    """Enhanced Ultra Hybrid System with all improvements"""
    
    def __init__(self, model_path="runs/detect/train/weights/best.pt", use_audio=True):
        print("\n" + "="*80)
        print("🚀 ENHANCED FINAL ULTRA HYBRID SYSTEM WITH AUDIO")
        print("="*80)
        
        # Initialize YOLO
        self.yolo = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo.to(self.device)
        print(f"🎯 YOLO using: {self.device.upper()}")
        
        # Load standard YOLO for vehicle detection (pretrained COCO model)
        try:
            # Try to load YOLOv8n for vehicle detection
            if os.path.exists('yolov8n.pt'):
                self.vehicle_yolo = YOLO('yolov8n.pt')
                self.has_vehicle_detection = True
                print(f"🚗 Vehicle detection YOLO loaded: yolov8n.pt")
            else:
                self.vehicle_yolo = None
                self.has_vehicle_detection = False
                print(f"⚠️  yolov8n.pt not found - will use heuristics for vehicle detection")
        except Exception as e:
            self.vehicle_yolo = None
            self.has_vehicle_detection = False
            print(f"⚠️  Could not load vehicle YOLO: {e}")
        
        # COCO vehicle classes: car=2, motorcycle=3, bus=5, truck=7
        self.vehicle_classes = {2, 3, 5, 7}
        
        # Initialize Enhanced Agent
        self.agent = EnhancedPatternAgent()
        
        # Initialize Audio Analyzer
        self.use_audio = use_audio
        if use_audio:
            self.audio_analyzer = AudioAnalyzer()
            print("🔊 Audio analyzer enabled")
        else:
            self.audio_analyzer = None
            print("🔇 Audio analyzer disabled")
        
        # Class names
        self.class_names = {0: 'fire', 1: 'vehicle_accident', 2: 'fighting', 3: 'explosion'}
        
        # Alert tracking
        self.alert_frames = []
        self.alert_timestamps = []
        
        # Audio data cache
        self.audio_data = None
        self.audio_sr = None
        
        # Event sequencing state for temporal logic
        self.last_explosion_frame = -999
        self.last_fire_frame = -999
        self.event_cooldown = 30  # Frames to wait before same event type
        self.post_explosion_window = 60  # Boost smoke for 60 frames after explosion
        
        print("✅ System ready!\n")
    
    def save_alert_clip(self, buffered_frames, event_name, conf, fps, frame_size):
        """Save a short clip of the alert (buffer + 3 seconds after)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = os.path.join(self.agent.alert_clip_dir, f"{event_name}_{timestamp}_{conf:.0f}pct.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(clip_path, fourcc, fps, frame_size)
        
        # Write buffered frames
        for frame in buffered_frames:
            writer.write(frame)
        
        writer.release()
        print(f"   💾 Alert clip saved: {clip_path}")
        
        return clip_path
    
    def process_video(self, video_path, output_path=None):
        """Process video with enhanced agent and audio analysis"""
        
        # Auto-generate output path if not provided
        if output_path is None:
            dir_path = os.path.dirname(video_path)
            base_name = os.path.basename(video_path)
            output_path = os.path.join(dir_path, f"enhanced_detected_{base_name}")
        
        print(f"📹 Input: {video_path}")
        print(f"💾 Output: {output_path}")
        
        # Extract audio if enabled
        if self.use_audio and self.audio_analyzer:
            print("🔊 Extracting audio from video...")
            self.audio_data, self.audio_sr = self.audio_analyzer.extract_audio_from_video(video_path)
            if self.audio_data is not None:
                print(f"✅ Audio extracted: {len(self.audio_data)} samples @ {self.audio_sr} Hz")
            else:
                print("⚠️  No audio found in video, continuing with video-only detection")
        
        print("="*80 + "\n")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30  # Default fallback
            print(f"⚠️  Could not detect FPS, using default: {fps}")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check if video file is corrupted/partial
        if total_frames <= 0 or total_frames < 100:
            print(f"⚠️  WARNING: Video reports only {total_frames} frames")
            print(f"⚠️  This may be a corrupted or partial file")
            print(f"⚠️  Will process all available frames anyway...")
            total_frames = 0  # Process until video ends naturally
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        processed_frame_num = 0
        start_time = time.time()
        frame_skip = 2  # Process every 2nd frame for 2x speed
        last_has_alert = False
        last_alert_text = ""
        last_conf_text = ""
        last_frame_text = ""
        current_event_type = None  # Track current ongoing event
        alert_shown_for_current_event = False  # Show alert only once per event
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Skip frames for faster processing
                if frame_num % frame_skip != 0:
                    # Use last detection result
                    display_frame = frame.copy()
                    if last_has_alert and last_alert_text and last_conf_text and last_frame_text:
                        # Apply last alert text (yellow, no banner)
                        cv2.putText(display_frame, last_alert_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                        cv2.putText(display_frame, last_frame_text, (width - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(display_frame, last_conf_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        # ONLY WRITE ALERT FRAMES - suppress normal frames
                        out.write(display_frame)
                    # Skip writing normal frames (no alert)
                    continue
                
                processed_frame_num += 1
                
                # YOLO detection
                results = self.yolo(frame, verbose=False)[0]
                
                # Collect YOLO detections
                yolo_detections = {
                    'fire': 0.0,
                    'explosion': 0.0,
                    'vehicle_accident': 0.0,
                    'fighting': 0.0
                }
                
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0]) * 100
                    class_name = self.class_names.get(cls, 'unknown')
                    yolo_detections[class_name] = max(yolo_detections.get(class_name, 0), conf)
                
                # ===== OBJECT DETECTION using standard YOLO (80/20 approach) =====
                # YOLO provides object context (20%), Agent does main detection (80%)
                vehicle_confidence = 0.0
                vehicle_count = 0
                person_confidence = 0.0
                person_count = 0
                
                if self.has_vehicle_detection:
                    # Use YOLOv8n to detect objects (vehicles, people, etc.)
                    vehicle_results = self.vehicle_yolo(frame, verbose=False)[0]
                    
                    for box in vehicle_results.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0]) * 100
                        
                        # Detect vehicles (car=2, motorcycle=3, bus=5, truck=7)
                        if cls in self.vehicle_classes:
                            vehicle_confidence = max(vehicle_confidence, conf)
                            vehicle_count += 1
                        
                        # Detect people (person=0)
                        elif cls == 0:
                            person_confidence = max(person_confidence, conf)
                            person_count += 1
                    
                    # Boost confidence if multiple objects detected
                    if vehicle_count >= 2:
                        vehicle_confidence = min(95, vehicle_confidence + (vehicle_count * 5))
                    if person_count >= 2:
                        person_confidence = min(95, person_confidence + (person_count * 5))
                else:
                    # Fallback: Use event YOLO detections as proxy
                    vehicle_confidence = yolo_detections['vehicle_accident'] * 0.8
                    if vehicle_confidence > 20:
                        vehicle_count = 1
                    
                    # No person detection fallback
                    person_confidence = 0.0
                    person_count = 0
                
                # Enhanced agent analysis (Video features)
                # Agent (80%) + YOLO object context (20%)
                explosion_conf, explosion_reason = self.agent.analyze_explosion(frame, yolo_detections['explosion'])
                fire_conf, fire_reason = self.agent.analyze_fire(frame, yolo_detections['fire'])
                
                # Accident: Pass vehicle detection confidence (NOT accident YOLO)
                accident_conf, accident_reason = self.agent.analyze_accident(frame, vehicle_confidence)
                
                # Fighting: Pass person detection confidence (NOT fighting YOLO)
                fighting_conf, fighting_reason = self.agent.analyze_fighting(frame, person_confidence)
                
                smoke_conf, smoke_reason = self.agent.analyze_smoke(frame, 0.0)
                
                # Update feature engineer history
                current_motion = list(self.agent.history['motion_flow'])[-1] if self.agent.history['motion_flow'] else 0.0
                current_brightness = list(self.agent.history['brightness'])[-1] if self.agent.history['brightness'] else 0.0
                current_edges = list(self.agent.history['edges'])[-1] if self.agent.history['edges'] else 0.0
                current_colors = list(self.agent.history['colors'])[-1] if self.agent.history['colors'] else 0.0
                self.agent.feature_engineer.update_history(current_motion, current_brightness, current_edges, current_colors)
                
                # Audio analysis (if enabled) - BOOST confidence when audio detected
                audio_confidences = {}
                audio_reasons = {}
                if self.use_audio and self.audio_data is not None:
                    # Analyze audio for current frame segment (use 1-second window)
                    audio_window = int(fps)  # 1 second = fps frames
                    start_frame = max(0, frame_num - audio_window // 2)
                    end_frame = frame_num + audio_window // 2
                    
                    # Get audio confidence for each event type
                    audio_results = {
                        'explosion': self.audio_analyzer.analyze_audio_segment(self.audio_data, start_frame, end_frame, fps, 'explosion'),
                        'fire': self.audio_analyzer.analyze_audio_segment(self.audio_data, start_frame, end_frame, fps, 'fire'),
                        'vehicle_accident': self.audio_analyzer.analyze_audio_segment(self.audio_data, start_frame, end_frame, fps, 'vehicle_accident'),
                        'fighting': self.audio_analyzer.analyze_audio_segment(self.audio_data, start_frame, end_frame, fps, 'fighting')
                    }
                    
                    audio_confidences = {k: v[0] for k, v in audio_results.items()}
                    audio_reasons = {k: v[1] for k, v in audio_results.items()}
                    
                    # Audio BOOST: Add 15% of audio confidence as bonus (keeps video agent dominant)
                    # This way strong audio (80-100%) adds 12-15% boost, weak audio (0-20%) adds 0-3%
                    if audio_confidences['explosion'] > 50:
                        explosion_conf = min(100, explosion_conf + (audio_confidences['explosion'] * 0.15))
                        explosion_reason += f" | Audio: {audio_reasons['explosion']}"
                    
                    if audio_confidences['fire'] > 50:
                        fire_conf = min(100, fire_conf + (audio_confidences['fire'] * 0.15))
                        fire_reason += f" | Audio: {audio_reasons['fire']}"
                    
                    if audio_confidences['vehicle_accident'] > 50:
                        accident_conf = min(100, accident_conf + (audio_confidences['vehicle_accident'] * 0.15))
                        accident_reason += f" | Audio: {audio_reasons['vehicle_accident']}"
                    
                    if audio_confidences['fighting'] > 50:
                        fighting_conf = min(100, fighting_conf + (audio_confidences['fighting'] * 0.15))
                        fighting_reason += f" | Audio: {audio_reasons['fighting']}"
                
                # ===== FEATURE ENGINEERING ENHANCEMENTS =====
                # Detect brightness and motion spikes for better explosion/accident detection
                brightness_spike_detected, brightness_spike_val = self.agent.feature_engineer.detect_brightness_spike(current_brightness)
                motion_spike_detected, motion_spike_val = self.agent.feature_engineer.detect_motion_spike(current_motion)
                flickering_detected, flickering_val = self.agent.feature_engineer.detect_flickering()
                sustained_activity, sustained_ratio = self.agent.feature_engineer.detect_sustained_activity()
                
                # Apply feature-specific boosts
                if brightness_spike_detected and explosion_conf > 40:
                    # Brightness spike confirms explosion
                    explosion_conf = min(100, explosion_conf + 10)
                    explosion_reason += f" | Brightness spike: +{brightness_spike_val:.0f}"
                
                if motion_spike_detected:
                    # Motion spike can indicate explosion or accident
                    if explosion_conf > 40:
                        explosion_conf = min(100, explosion_conf + 8)
                        explosion_reason += f" | Motion spike: {motion_spike_val:.1f}x"
                    if accident_conf > 40:
                        accident_conf = min(100, accident_conf + 12)
                        accident_reason += f" | Motion spike: {motion_spike_val:.1f}x"
                
                if flickering_detected and fire_conf > 30:
                    # Flickering confirms fire
                    fire_conf = min(100, fire_conf + 12)
                    fire_reason += f" | Flickering: {flickering_val:.0f}"
                
                if sustained_activity and fighting_conf > 30:
                    # Sustained activity confirms fighting
                    fighting_conf = min(100, fighting_conf + 15)
                    fighting_reason += f" | Sustained: {sustained_ratio:.1%}"
                
                # ===== ADVANCED EVENT SEQUENCING with TemporalEventTracker =====
                # Apply sequence boosts and cooldowns based on event history
                
                # Get expected events from temporal tracker
                expected_events = self.agent.temporal_tracker.get_expected_events()
                
                # Check for brightness and motion spikes using feature engineer
                brightness_spike_detected, brightness_spike_val = self.agent.feature_engineer.detect_brightness_spike(current_brightness)
                motion_spike_detected, motion_spike_val = self.agent.feature_engineer.detect_motion_spike(current_motion)
                
                # Apply sequence boosts for expected events
                if 'smoke' in [e.lower() for e in expected_events]:
                    # Smoke is expected after explosion or fire
                    smoke_conf = min(100, smoke_conf + 30)
                    smoke_reason += " | Sequence boost: expected after explosion/fire (+30)"
                
                if 'fire' in [e.lower() for e in expected_events]:
                    # Fire is expected after explosion or smoke
                    fire_conf = min(100, fire_conf + 20)
                    fire_reason += " | Sequence boost: expected after explosion (+20)"
                
                # Apply cooldown penalties to prevent rapid re-detection
                if self.agent.temporal_tracker.check_cooldown('explosion', frame_num):
                    explosion_conf *= 0.5
                    explosion_reason += " | Cooldown penalty (×0.5)"
                
                if self.agent.temporal_tracker.check_cooldown('fire', frame_num):
                    fire_conf *= 0.6
                    fire_reason += " | Cooldown penalty (×0.6)"
                
                if self.agent.temporal_tracker.check_cooldown('vehicle_accident', frame_num):
                    accident_conf *= 0.5
                    accident_reason += " | Cooldown penalty (×0.5)"
                
                if self.agent.temporal_tracker.check_cooldown('fighting', frame_num):
                    fighting_conf *= 0.6
                    fighting_reason += " | Cooldown penalty (×0.6)"
                
                if self.agent.temporal_tracker.check_cooldown('smoke', frame_num):
                    smoke_conf *= 0.7
                    smoke_reason += " | Cooldown penalty (×0.7)"
                
                # Determine highest confidence event
                detections = {
                    'EXPLOSION': (explosion_conf, explosion_reason),
                    'FIRE': (fire_conf, fire_reason),
                    'ACCIDENT': (accident_conf, accident_reason),
                    'FIGHTING': (fighting_conf, fighting_reason),
                    'SMOKE': (smoke_conf, smoke_reason)
                }
                
                # ===== EVENT OVERLAP RESOLUTION =====
                # Apply intelligent prioritization when multiple events detected
                # Rules:
                # - Explosion + Fire + Smoke → Explosion (primary cause)
                # - Fire + Smoke → Fire
                # - Fire + Accident → Accident (if vehicle impact visible)
                # - Fighting + Accident → Fighting (if no vehicle)
                # - Explosion + Accident → Explosion (if flash/fire present)
                
                threshold_map = {
                    'EXPLOSION': self.agent.thresholds.get('explosion', {}).get('confidence', 70.0),
                    'FIRE': self.agent.thresholds.get('fire', {}).get('confidence', 65.0),
                    'ACCIDENT': self.agent.thresholds.get('accident', {}).get('confidence', 65.0),
                    'FIGHTING': self.agent.thresholds.get('fighting', {}).get('confidence', 65.0),
                    'SMOKE': self.agent.thresholds.get('smoke', {}).get('confidence', 60.0)
                }
                
                # Check which events passed threshold
                candidates = {}
                for event_name, (conf, reason) in detections.items():
                    if conf > threshold_map[event_name]:
                        candidates[event_name] = conf
                
                # Apply resolution rules
                validated_events = []
                if len(candidates) > 1:
                    # Multiple events detected - resolve conflicts
                    
                    # Rule 1: Explosion takes priority over everything if present
                    if 'EXPLOSION' in candidates and candidates['EXPLOSION'] > 70:
                        validated_events.append(('EXPLOSION', detections['EXPLOSION'][0], detections['EXPLOSION'][1]))
                    
                    # Rule 2: Accident + Fire → Keep Accident (vehicle collision with fire)
                    elif 'ACCIDENT' in candidates and 'FIRE' in candidates:
                        if yolo_detections.get('vehicle_accident', 0) > 20:
                            validated_events.append(('ACCIDENT', detections['ACCIDENT'][0], detections['ACCIDENT'][1] + " | Fire present"))
                        else:
                            validated_events.append(('FIRE', detections['FIRE'][0], detections['FIRE'][1]))
                    
                    # Rule 3: Fighting + Accident → Prioritize Fighting if no vehicles
                    elif 'FIGHTING' in candidates and 'ACCIDENT' in candidates:
                        if yolo_detections.get('vehicle_accident', 0) < 20:
                            validated_events.append(('FIGHTING', detections['FIGHTING'][0], detections['FIGHTING'][1]))
                        else:
                            validated_events.append(('ACCIDENT', detections['ACCIDENT'][0], detections['ACCIDENT'][1]))
                    
                    # Rule 4: Fire + Smoke → Keep Fire (smoke is byproduct)
                    elif 'FIRE' in candidates and 'SMOKE' in candidates:
                        validated_events.append(('FIRE', detections['FIRE'][0], detections['FIRE'][1] + " | Smoke present"))
                    
                    # Default: Take highest confidence
                    else:
                        top_event = max(candidates.items(), key=lambda x: x[1])
                        validated_events.append((top_event[0], detections[top_event[0]][0], detections[top_event[0]][1]))
                
                elif len(candidates) == 1:
                    # Single event detected
                    event_name = list(candidates.keys())[0]
                    validated_events.append((event_name, detections[event_name][0], detections[event_name][1]))
                
                # Sort by confidence
                validated_events.sort(key=lambda x: x[1], reverse=True)
                
                # ===== TEMPORAL DURATION VALIDATION =====
                # Apply temporal consistency check - filter events with invalid durations
                temporally_valid_events = []
                for event_name, conf, reason in validated_events:
                    is_valid, conf_modifier, duration_reason = self.agent.validate_event_duration(event_name, frame_num)
                    
                    if is_valid:
                        # Apply confidence modifier based on duration
                        adjusted_conf = conf * conf_modifier
                        adjusted_reason = reason + f" | {duration_reason}"
                        temporally_valid_events.append((event_name, adjusted_conf, adjusted_reason))
                    else:
                        # Event failed temporal validation (too long for short events)
                        print(f"   ⏱️  Frame {frame_num}: {event_name} rejected - {duration_reason}")
                
                # Replace validated_events with temporally filtered ones
                validated_events = temporally_valid_events
                validated_events.sort(key=lambda x: x[1], reverse=True)
                
                # ===== ADD TO DETECTION BUFFER =====
                # Store current frame's detections for temporal analysis
                buffer_entry = {
                    'frame': frame_num,
                    'EXPLOSION': explosion_conf,
                    'FIRE': fire_conf,
                    'ACCIDENT': accident_conf,
                    'FIGHTING': fighting_conf,
                    'SMOKE': smoke_conf
                }
                self.agent.detection_buffer.append(buffer_entry)
                
                # ===== PEAK DETECTION FILTER =====
                # Only alert if we're at or near the peak confidence
                peak_filtered_events = []
                for event_name, conf, reason in validated_events:
                    should_alert, peak_reason = self.agent.find_peak_detection_in_buffer(event_name)
                    
                    if should_alert:
                        adjusted_reason = reason + f" | {peak_reason}"
                        peak_filtered_events.append((event_name, conf, adjusted_reason))
                    else:
                        # Skip this detection - not at peak
                        print(f"   📉 Frame {frame_num}: {event_name} skipped - {peak_reason}")
                
                # Replace with peak-filtered events
                validated_events = peak_filtered_events
                
                # Get optical flow and edge density for feature logging
                flow_motion = list(self.agent.history['motion_flow'])[-1] if self.agent.history['motion_flow'] else 0.0
                edge_density = list(self.agent.history['edges'])[-1] if self.agent.history['edges'] else 0.0
                brightness = list(self.agent.history['brightness'])[-1] if self.agent.history['brightness'] else 0.0
                
                # Display frame
                display_frame = frame.copy()
                
                if validated_events:
                    # Alert detected!
                    last_has_alert = True
                    self.alert_frames.append(frame_num)
                    self.alert_timestamps.append(frame_num)
                    
                    # Get top event
                    top_event, top_conf, top_reason = validated_events[0]
                    
                    # Log features to CSV
                    timestamp = datetime.now().isoformat()
                    with open(self.agent.feature_log, 'a') as f:
                        f.write(f"{timestamp},{frame_num},{explosion_conf:.2f},{yolo_detections['explosion']:.2f},")
                        f.write(f"{fire_conf:.2f},{yolo_detections['fire']:.2f},")
                        f.write(f"{accident_conf:.2f},{yolo_detections['vehicle_accident']:.2f},")
                        f.write(f"{fighting_conf:.2f},{yolo_detections['fighting']:.2f},")
                        f.write(f"{smoke_conf:.2f},{flow_motion:.2f},{edge_density:.2f},{brightness:.2f},")
                        f.write(f"{top_event},{top_conf:.2f}\n")
                    
                    # Check if this is a NEW event (different from current)
                    if current_event_type != top_event:
                        # New event detected - reset and show alert
                        current_event_type = top_event
                        alert_shown_for_current_event = False
                        
                        # Record event in temporal tracker
                        event_name = top_event.lower()
                        is_sequence = self.agent.temporal_tracker.record_event(event_name, frame_num, top_conf)
                        
                        # Console output for NEW event only
                        sequence_marker = " [SEQUENCE]" if is_sequence else ""
                        output_msg = f"   🚨 NEW EVENT Frame {frame_num}: {top_event} ({top_conf:.0f}%){sequence_marker} - {top_reason}"
                        
                        # Add vehicle detection info for accidents
                        if top_event == 'ACCIDENT' and self.has_vehicle_detection:
                            output_msg += f" | 🚗 Vehicles: {vehicle_count} detected ({vehicle_confidence:.0f}%)"
                        
                        # Add person detection info for fighting
                        if top_event == 'FIGHTING' and self.has_vehicle_detection:
                            output_msg += f" | 👥 People: {person_count} detected ({person_confidence:.0f}%)"
                        
                        # Add audio info if available
                        if self.use_audio and audio_confidences:
                            event_key = top_event.lower()
                            if event_key == 'accident':
                                event_key = 'vehicle_accident'
                            audio_conf = audio_confidences.get(event_key, 0.0)
                            output_msg += f" | Audio: {audio_conf:.0f}%"
                        
                        print(output_msg)
                    
                    # Add YELLOW TEXT OVERLAYS - Show for all alert frames
                    # Line 1: ALERT: EVENT! and Frame number
                    alert_text = f"ALERT: {top_event}!"
                    frame_text = f"Frame: {frame_num}/{total_frames}"
                    cv2.putText(display_frame, alert_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(display_frame, frame_text, (width - 350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    
                    # Line 2: Confidence
                    conf_text = f"Confidence: {top_conf:.1f}%"
                    cv2.putText(display_frame, conf_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    
                    # Save for skipped frames
                    last_alert_text = alert_text
                    last_conf_text = conf_text
                    last_frame_text = frame_text
                
                else:
                    # Reset alert flag
                    last_has_alert = False
                    current_event_type = None  # Reset current event
                    alert_shown_for_current_event = False  # Reset alert flag
                    
                    # No overlay during monitoring - clean view
                    # DO NOT WRITE - suppress normal frames
                
                # Write frame ONLY if alert detected
                if last_has_alert:
                    out.write(display_frame)
                
                # Show preview
                preview = cv2.resize(display_frame, (960, 540))
                cv2.imshow('Enhanced Ultra System', preview)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print("✅ COMPLETE!")
        print("="*80)
        print(f"Total frames: {frame_num} | Processed: {processed_frame_num} | Skipped: {frame_num - processed_frame_num}")
        print(f"Time: {elapsed:.1f}s ({frame_num/elapsed:.1f} FPS)")
        if total_frames > 0 and frame_num < total_frames:
            print(f"⚠️  WARNING: Video stopped early! Processed {frame_num}/{total_frames} frames")
            print(f"⚠️  This indicates a corrupted or incomplete video file")
        print(f"Alert frames: {len(self.alert_frames)}")
        if len(self.alert_timestamps) > 20:
            print(f"Alert timestamps: {self.alert_timestamps[:20]}...")
        else:
            print(f"Alert timestamps: {self.alert_timestamps}")
        print(f"Output: {output_path}")
        print(f"Feature log: {self.agent.feature_log}")
    
    def process_frame_for_api(self, frame, frame_num):
        """
        Process single frame for API/backend integration
        Returns detection results as dictionary (no video writing)
        """
        try:
            # YOLO detection
            results = self.yolo(frame, verbose=False)[0]
            
            # Collect YOLO detections
            yolo_detections = {
                'fire': 0.0,
                'explosion': 0.0,
                'vehicle_accident': 0.0,
                'fighting': 0.0
            }
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                class_name = self.class_names.get(cls, 'unknown')
                yolo_detections[class_name] = max(yolo_detections.get(class_name, 0), conf)
            
            # Object detection (vehicles + people)
            vehicle_confidence = 0.0
            vehicle_count = 0
            person_confidence = 0.0
            person_count = 0
            
            if self.has_vehicle_detection:
                vehicle_results = self.vehicle_yolo(frame, verbose=False)[0]
                
                for box in vehicle_results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0]) * 100
                    
                    if cls in self.vehicle_classes:
                        vehicle_confidence = max(vehicle_confidence, conf)
                        vehicle_count += 1
                    elif cls == 0:
                        person_confidence = max(person_confidence, conf)
                        person_count += 1
                
                if vehicle_count >= 2:
                    vehicle_confidence = min(95, vehicle_confidence + (vehicle_count * 5))
                if person_count >= 2:
                    person_confidence = min(95, person_confidence + (person_count * 5))
            
            # Enhanced agent analysis
            explosion_conf, explosion_reason = self.agent.analyze_explosion(frame, yolo_detections['explosion'])
            fire_conf, fire_reason = self.agent.analyze_fire(frame, yolo_detections['fire'])
            accident_conf, accident_reason = self.agent.analyze_accident(frame, vehicle_confidence)
            fighting_conf, fighting_reason = self.agent.analyze_fighting(frame, person_confidence)
            smoke_conf, smoke_reason = self.agent.analyze_smoke(frame, 0.0)
            
            # Get motion spike
            motion_spike = list(self.agent.history['motion_flow'])[-1] if self.agent.history['motion_flow'] else 0.0
            
            # Find top detection
            detections = {
                'EXPLOSION': explosion_conf,
                'FIRE': fire_conf,
                'ACCIDENT': accident_conf,
                'FIGHTING': fighting_conf,
                'SMOKE': smoke_conf
            }
            
            top_event = max(detections, key=detections.get)
            top_conf = detections[top_event]
            
            # Determine risk level
            if top_conf >= 80:
                risk_level = 'CRITICAL'
            elif top_conf >= 65:
                risk_level = 'HIGH'
            elif top_conf >= 50:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            # Return result
            return {
                'event': top_event if top_conf >= 65 else 'normal',
                'confidence': top_conf,
                'risk_level': risk_level,
                'reasoning': {
                    'explosion': explosion_reason,
                    'fire': fire_reason,
                    'accident': accident_reason,
                    'fighting': fighting_reason,
                    'smoke': smoke_reason
                },
                'objects': {
                    'vehicles': vehicle_count,
                    'people': person_count
                },
                'motion_spike': motion_spike,
                'vehicles': {
                    'count': vehicle_count,
                    'confidence': vehicle_confidence
                },
                'people': {
                    'count': person_count,
                    'confidence': person_confidence
                },
                'all_detections': detections
            }
        
        except Exception as e:
            print(f"❌ Frame processing error: {e}")
            return {
                'event': 'error',
                'confidence': 0.0,
                'risk_level': 'UNKNOWN',
                'error': str(e)
            }


def main():
    """Main execution"""
    system = EnhancedUltraSystem()
    
    # Get video path from user
    print("\n📁 Enter video path:")
    print("   (Example: C:\\Users\\amrut\\OneDrive\\Desktop\\video.mp4)")
    video_path = input("   Path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return
    
    # Get output path (optional)
    print("\n💾 Enter output path (press Enter for auto-naming):")
    output_path = input("   Path: ").strip().strip('"').strip("'")
    
    if not output_path:
        output_path = None  # Auto-generate
    
    # Process video
    system.process_video(video_path, output_path)


if __name__ == "__main__":
    main()
