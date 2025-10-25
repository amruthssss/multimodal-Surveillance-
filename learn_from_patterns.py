"""
PATTERN ANALYSIS & LEARNING SYSTEM
Analyzes collected patterns to improve detection thresholds
"""

import pickle
import numpy as np
from collections import defaultdict
import json

class PatternAnalyzer:
    """Analyzes collected patterns to learn optimal detection parameters"""
    
    def __init__(self):
        self.patterns = []
        self.analysis = {}
        
    def load_patterns(self):
        """Load all collected patterns"""
        pattern_files = [
            'models/web_learned_patterns_extended.pkl',
            'models/web_learned_knowledge.pkl',
            'models/pure_agent_knowledge.pkl',
            'models/agent_knowledge.pkl',
            'models/expanded_patterns.pkl'
        ]
        
        total = 0
        for file in pattern_files:
            try:
                with open(file, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        self.patterns.extend(data)
                        total += len(data)
                    elif isinstance(data, dict):
                        for key, val in data.items():
                            if isinstance(val, list):
                                self.patterns.extend(val)
                                total += len(val)
                print(f"✅ Loaded: {file} ({total} patterns)")
            except Exception as e:
                print(f"⚠️  Skipped: {file} ({e})")
        
        print(f"\n📊 Total patterns loaded: {len(self.patterns)}")
        return len(self.patterns)
    
    def analyze_event_patterns(self, event_type):
        """Analyze patterns for specific event type"""
        event_patterns = [p for p in self.patterns if isinstance(p, dict) and p.get('event_type') == event_type]
        
        if not event_patterns:
            return None
        
        print(f"\n📈 Analyzing {event_type} patterns ({len(event_patterns)} samples)...")
        
        # Extract features
        features = defaultdict(list)
        
        for pattern in event_patterns:
            if 'features' in pattern:
                feat = pattern['features']
                
                # Color features
                if 'color_histogram' in feat:
                    features['brightness'].append(np.mean(feat['color_histogram']))
                
                if 'hsv_features' in feat:
                    features['hue'].append(feat['hsv_features'].get('mean_hue', 0))
                    features['saturation'].append(feat['hsv_features'].get('mean_saturation', 0))
                    features['value'].append(feat['hsv_features'].get('mean_value', 0))
                
                # Edge features
                if 'edge_density' in feat:
                    features['edges'].append(feat['edge_density'])
                
                # Texture features
                if 'texture_variance' in feat:
                    features['texture'].append(feat['texture_variance'])
        
        # Calculate statistics
        stats = {}
        for feature_name, values in features.items():
            if values:
                stats[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'p25': np.percentile(values, 25),
                    'p75': np.percentile(values, 75)
                }
                print(f"   {feature_name}: mean={stats[feature_name]['mean']:.2f}, "
                      f"range=[{stats[feature_name]['min']:.2f}, {stats[feature_name]['max']:.2f}]")
        
        return stats
    
    def generate_optimal_thresholds(self):
        """Generate optimal detection thresholds based on learned patterns"""
        print("\n" + "="*80)
        print("🎯 GENERATING OPTIMAL THRESHOLDS FROM LEARNED PATTERNS")
        print("="*80)
        
        optimal_thresholds = {}
        
        for event_type in ['explosion', 'fire', 'vehicle_accident', 'fighting', 'smoke']:
            stats = self.analyze_event_patterns(event_type)
            
            if stats:
                # Generate thresholds based on statistics
                thresholds = {}
                
                # Brightness threshold (use median)
                if 'brightness' in stats:
                    thresholds['brightness_min'] = stats['brightness']['median']
                
                # Edge density (use 25th percentile to catch most cases)
                if 'edges' in stats:
                    thresholds['edge_density_min'] = stats['edges']['p25']
                
                # Texture variance (use 25th percentile)
                if 'texture' in stats:
                    thresholds['texture_variance'] = stats['texture']['p25']
                
                optimal_thresholds[event_type] = thresholds
                
                print(f"\n✅ {event_type.upper()} Optimal Thresholds:")
                for key, val in thresholds.items():
                    print(f"   {key}: {val:.2f}")
        
        return optimal_thresholds
    
    def save_learned_thresholds(self, thresholds):
        """Save learned thresholds to file"""
        output_file = 'models/learned_optimal_thresholds.json'
        with open(output_file, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"\n💾 Saved optimal thresholds to: {output_file}")
    
    def compare_with_current(self, current_thresholds):
        """Compare learned thresholds with current settings"""
        print("\n" + "="*80)
        print("📊 COMPARISON: Current vs Learned Optimal")
        print("="*80)
        
        optimal = self.generate_optimal_thresholds()
        
        for event_type in optimal.keys():
            print(f"\n{event_type.upper()}:")
            
            if event_type in current_thresholds:
                current = current_thresholds[event_type]
                learned = optimal[event_type]
                
                for key in learned.keys():
                    if key in current:
                        current_val = current[key]
                        learned_val = learned[key]
                        diff = learned_val - current_val
                        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
                        print(f"   {key}:")
                        print(f"      Current: {current_val:.2f}")
                        print(f"      Learned: {learned_val:.2f} {arrow}")


def main():
    """Main learning process"""
    print("="*80)
    print("🧠 PATTERN LEARNING & THRESHOLD OPTIMIZATION")
    print("="*80)
    print("Analyzing collected patterns to learn optimal detection thresholds...")
    print()
    
    analyzer = PatternAnalyzer()
    
    # Load all patterns
    total_patterns = analyzer.load_patterns()
    
    if total_patterns == 0:
        print("\n❌ No patterns found to analyze!")
        return
    
    # Generate optimal thresholds
    optimal_thresholds = analyzer.generate_optimal_thresholds()
    
    # Save results
    analyzer.save_learned_thresholds(optimal_thresholds)
    
    print("\n" + "="*80)
    print("✅ LEARNING COMPLETE!")
    print("="*80)
    print(f"Analyzed {total_patterns} patterns")
    print(f"Generated optimal thresholds for {len(optimal_thresholds)} event types")
    print("\nTo use these thresholds, update enhanced_final_ultra_system.py")
    print("with the values from models/learned_optimal_thresholds.json")
    print()


if __name__ == "__main__":
    main()
