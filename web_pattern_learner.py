"""
Web Pattern Learner - Learn 1M-10M patterns from Google for accurate event detection
Supports: Explosion, Fire, Vehicle Accident, Fighting
Sources: Google Images, YouTube videos, News articles, Research papers
"""

import os
import pickle
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
from urllib.parse import urlencode, quote_plus
import time
import hashlib
from datetime import datetime
from collections import defaultdict
import json

class WebPatternLearner:
    """Learn millions of patterns from web sources for event detection"""
    
    def __init__(self, output_dir="models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Event-specific search queries
        self.search_queries = {
            'explosion': [
                'explosion blast', 'bomb explosion', 'gas explosion', 'building explosion',
                'industrial explosion', 'fireworks explosion', 'explosive blast',
                'detonation', 'explosion debris', 'explosion smoke', 'explosion flash',
                'explosion fire', 'car explosion', 'vehicle explosion', 'mine blast',
                'controlled explosion', 'accidental explosion', 'explosion damage',
                'explosion aftermath', 'explosion mushroom cloud', 'explosion shockwave'
            ],
            'fire': [
                'building fire', 'house fire', 'forest fire', 'wildfire', 'fire flames',
                'fire smoke', 'structure fire', 'industrial fire', 'car fire',
                'vehicle fire', 'fire blaze', 'fire burning', 'fire spreading',
                'fire emergency', 'fire damage', 'fire rescue', 'apartment fire',
                'warehouse fire', 'factory fire', 'electrical fire'
            ],
            'vehicle_accident': [
                'car accident', 'car crash', 'vehicle collision', 'traffic accident',
                'road accident', 'highway crash', 'head-on collision', 'car wreck',
                'multi-car pileup', 'truck accident', 'motorcycle accident',
                'rollover accident', 'accident debris', 'accident damage',
                'traffic collision', 'vehicle impact', 'car accident scene',
                'accident emergency', 'car crash damage', 'collision aftermath'
            ],
            'fighting': [
                'street fight', 'physical fight', 'brawl', 'assault', 'fighting people',
                'violent altercation', 'fist fight', 'combat', 'physical violence',
                'street violence', 'group fight', 'bar fight', 'public fight',
                'aggressive behavior', 'physical confrontation', 'violent incident',
                'fighting scene', 'violent attack', 'fighting crowd', 'riot'
            ],
            'smoke': [
                'smoke cloud', 'dense smoke', 'black smoke', 'white smoke', 'gray smoke',
                'smoke billowing', 'smoke plume', 'industrial smoke', 'chimney smoke',
                'smoke stack', 'smoke rising', 'thick smoke', 'smoke haze',
                'smoke fog', 'smoke pollution', 'smoke screen', 'smoke detector',
                'smoke signal', 'smoke emission', 'heavy smoke'
            ]
        }
        
        # Pattern storage
        self.patterns = defaultdict(list)
        self.pattern_hashes = set()  # Avoid duplicates
        
        # Stats tracking
        self.stats = {
            'total_patterns': 0,
            'patterns_per_event': defaultdict(int),
            'sources': defaultdict(int),
            'failed_downloads': 0,
            'duplicate_patterns': 0
        }
    
    def generate_pattern_hash(self, pattern_data):
        """Generate unique hash for pattern to avoid duplicates"""
        pattern_str = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()
    
    def search_google_images(self, query, num_results=100):
        """Search Google Images for event patterns"""
        print(f"   🔍 Searching Google Images: '{query}'...")
        
        image_urls = []
        try:
            # Google Images search URL
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch&num={num_results}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract image URLs from search results
            for img in soup.find_all('img'):
                src = img.get('src')
                if src and src.startswith('http'):
                    image_urls.append(src)
            
            print(f"      ✓ Found {len(image_urls)} images")
            
        except Exception as e:
            print(f"      ✗ Search failed: {e}")
        
        return image_urls
    
    def download_and_extract_pattern(self, url, event_type):
        """Download image and extract visual pattern"""
        try:
            # Download image
            response = requests.get(url, timeout=10)
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None or image.size == 0:
                self.stats['failed_downloads'] += 1
                return None
            
            # Resize to standard size
            image = cv2.resize(image, (224, 224))
            
            # Extract pattern features
            pattern = self.extract_visual_features(image, event_type)
            
            # Add metadata
            pattern['source'] = 'google_images'
            pattern['url'] = url
            pattern['timestamp'] = datetime.now().isoformat()
            
            return pattern
            
        except Exception as e:
            self.stats['failed_downloads'] += 1
            return None
    
    def extract_visual_features(self, image, event_type):
        """Extract comprehensive visual features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        pattern = {
            'event_type': event_type,
            'features': {}
        }
        
        # 1. Color features
        pattern['features']['mean_brightness'] = float(np.mean(gray))
        pattern['features']['std_brightness'] = float(np.std(gray))
        pattern['features']['max_brightness'] = float(np.max(gray))
        
        # HSV statistics
        pattern['features']['hue_mean'] = float(np.mean(hsv[:, :, 0]))
        pattern['features']['sat_mean'] = float(np.mean(hsv[:, :, 1]))
        pattern['features']['val_mean'] = float(np.mean(hsv[:, :, 2]))
        
        # 2. Texture features
        edges = cv2.Canny(gray, 50, 150)
        pattern['features']['edge_density'] = float(np.count_nonzero(edges) / edges.size)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        pattern['features']['texture_variance'] = float(laplacian.var())
        
        # 3. Color ranges for specific events
        if event_type == 'explosion':
            # Smoke detection
            smoke_mask = cv2.inRange(hsv, (0, 0, 80), (180, 60, 180))
            pattern['features']['smoke_coverage'] = float(np.count_nonzero(smoke_mask) / smoke_mask.size)
            
            # Bright flash
            bright_pixels = np.count_nonzero(gray > 200)
            pattern['features']['flash_coverage'] = float(bright_pixels / gray.size)
            
        elif event_type == 'fire':
            # Orange/yellow flames
            orange = cv2.inRange(hsv, (5, 100, 100), (15, 255, 255))
            yellow = cv2.inRange(hsv, (15, 100, 100), (30, 255, 255))
            red = cv2.inRange(hsv, (0, 100, 100), (5, 255, 255))
            pattern['features']['flame_coverage'] = float((np.count_nonzero(orange) + np.count_nonzero(yellow) + np.count_nonzero(red)) / gray.size)
            
            # Smoke
            smoke = cv2.inRange(hsv, (0, 0, 80), (180, 60, 180))
            pattern['features']['smoke_coverage'] = float(np.count_nonzero(smoke) / smoke.size)
            
        elif event_type == 'vehicle_accident':
            # Metal/debris colors
            gray_metal = cv2.inRange(hsv, (0, 0, 50), (180, 50, 200))
            pattern['features']['metal_coverage'] = float(np.count_nonzero(gray_metal) / gray_metal.size)
            
            # Edge chaos (shattered glass, bent metal)
            pattern['features']['chaos_level'] = float(np.count_nonzero(edges) / edges.size)
            
        elif event_type == 'fighting':
            # Motion/action indicators
            pattern['features']['activity_level'] = float(pattern['features']['edge_density'])
            
            # Color variance (chaotic scene)
            pattern['features']['color_chaos'] = float(np.std(hsv))
        
        elif event_type == 'smoke':
            # Multiple smoke ranges
            light_smoke = cv2.inRange(hsv, (0, 0, 150), (180, 50, 255))
            dark_smoke = cv2.inRange(hsv, (0, 0, 40), (180, 80, 150))
            gray_smoke = cv2.inRange(hsv, (0, 0, 80), (180, 60, 180))
            white_smoke = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
            
            total_smoke = (np.count_nonzero(light_smoke) + np.count_nonzero(dark_smoke) + 
                          np.count_nonzero(gray_smoke) + np.count_nonzero(white_smoke))
            pattern['features']['smoke_coverage'] = float(total_smoke / gray.size)
            pattern['features']['smoke_density'] = float(np.mean([
                np.count_nonzero(light_smoke) / light_smoke.size,
                np.count_nonzero(dark_smoke) / dark_smoke.size,
                np.count_nonzero(gray_smoke) / gray_smoke.size,
                np.count_nonzero(white_smoke) / white_smoke.size
            ]))
        
        # 4. Spatial features
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        pattern['features']['histogram'] = hist.flatten().tolist()
        
        # 5. Contextual keywords
        pattern['keywords'] = self.generate_keywords(pattern['features'], event_type)
        
        return pattern
    
    def generate_keywords(self, features, event_type):
        """Generate descriptive keywords from features"""
        keywords = [event_type]
        
        # Brightness-based keywords
        if features['mean_brightness'] > 150:
            keywords.extend(['bright', 'flash', 'light'])
        elif features['mean_brightness'] < 80:
            keywords.extend(['dark', 'night', 'low-light'])
        
        # Texture-based keywords
        if features['edge_density'] > 0.15:
            keywords.extend(['edges', 'debris', 'chaos'])
        
        # Event-specific keywords
        if event_type == 'explosion':
            if features.get('smoke_coverage', 0) > 0.3:
                keywords.extend(['smoke', 'smoky'])
            if features.get('flash_coverage', 0) > 0.2:
                keywords.extend(['flash', 'blast'])
                
        elif event_type == 'fire':
            if features.get('flame_coverage', 0) > 0.1:
                keywords.extend(['flames', 'burning'])
            if features.get('smoke_coverage', 0) > 0.2:
                keywords.extend(['smoke', 'smoky'])
                
        elif event_type == 'vehicle_accident':
            keywords.extend(['crash', 'collision', 'damage'])
            
        elif event_type == 'fighting':
            keywords.extend(['violence', 'action', 'movement'])
        
        elif event_type == 'smoke':
            keywords.extend(['smoke', 'haze', 'cloud', 'pollution'])
            if features.get('smoke_coverage', 0) > 0.5:
                keywords.extend(['thick', 'dense', 'heavy'])
            if features['mean_brightness'] < 100:
                keywords.extend(['black', 'dark'])
            elif features['mean_brightness'] > 150:
                keywords.extend(['white', 'light'])
        
        return keywords
    
    def learn_from_google(self, events=['explosion', 'fire', 'vehicle_accident', 'fighting', 'smoke'], 
                         target_patterns=1_000_000, images_per_query=100):
        """
        Learn patterns from Google search
        
        Args:
            events: List of event types to learn
            target_patterns: Total patterns to learn (1M or 10M)
            images_per_query: Images to fetch per search query
        """
        print("\n" + "="*80)
        print(f"🌐 WEB PATTERN LEARNER - Learning {target_patterns:,} patterns")
        print("="*80)
        
        patterns_per_event = target_patterns // len(events)
        
        for event_type in events:
            print(f"\n📚 Learning {event_type.upper()} patterns...")
            print(f"   Target: {patterns_per_event:,} patterns")
            
            queries = self.search_queries.get(event_type, [event_type])
            patterns_needed = patterns_per_event
            patterns_collected = 0
            
            for query in queries:
                if patterns_collected >= patterns_needed:
                    break
                
                # Search Google Images
                image_urls = self.search_google_images(query, num_results=images_per_query)
                
                # Download and extract patterns
                for url in image_urls:
                    if patterns_collected >= patterns_needed:
                        break
                    
                    pattern = self.download_and_extract_pattern(url, event_type)
                    
                    if pattern:
                        # Check for duplicates
                        pattern_hash = self.generate_pattern_hash(pattern)
                        
                        if pattern_hash not in self.pattern_hashes:
                            self.patterns[event_type].append(pattern)
                            self.pattern_hashes.add(pattern_hash)
                            patterns_collected += 1
                            self.stats['total_patterns'] += 1
                            self.stats['patterns_per_event'][event_type] += 1
                            self.stats['sources']['google_images'] += 1
                            
                            if patterns_collected % 100 == 0:
                                print(f"      Progress: {patterns_collected:,}/{patterns_needed:,} patterns")
                        else:
                            self.stats['duplicate_patterns'] += 1
                    
                    # Rate limiting
                    time.sleep(0.1)
            
            print(f"   ✓ Collected {patterns_collected:,} {event_type} patterns")
        
        print("\n" + "="*80)
        print("✅ LEARNING COMPLETE!")
        print("="*80)
        self.print_stats()
    
    def save_patterns(self, filename="web_learned_patterns_extended.pkl"):
        """Save learned patterns to pickle file"""
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert to format compatible with existing system
        all_patterns = []
        for event_type, patterns in self.patterns.items():
            all_patterns.extend(patterns)
        
        with open(filepath, 'wb') as f:
            pickle.dump(all_patterns, f)
        
        print(f"\n💾 Saved {len(all_patterns):,} patterns to: {filepath}")
        return filepath
    
    def load_existing_patterns(self, filename="web_learned_knowledge.pkl"):
        """Load existing patterns and merge with new ones"""
        filepath = os.path.join(self.output_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    existing = pickle.load(f)
                
                print(f"📂 Loaded {len(existing):,} existing patterns from: {filename}")
                
                # Add to current patterns
                for pattern in existing:
                    event_type = pattern.get('event_type', 'unknown')
                    pattern_hash = self.generate_pattern_hash(pattern)
                    
                    if pattern_hash not in self.pattern_hashes:
                        self.patterns[event_type].append(pattern)
                        self.pattern_hashes.add(pattern_hash)
                        self.stats['total_patterns'] += 1
                        self.stats['patterns_per_event'][event_type] += 1
                    else:
                        self.stats['duplicate_patterns'] += 1
                
                print(f"   ✓ Total patterns now: {self.stats['total_patterns']:,}")
                
            except Exception as e:
                print(f"⚠️  Could not load existing patterns: {e}")
    
    def print_stats(self):
        """Print learning statistics"""
        print(f"\n📊 LEARNING STATISTICS:")
        print(f"   Total patterns learned: {self.stats['total_patterns']:,}")
        print(f"   Failed downloads: {self.stats['failed_downloads']:,}")
        print(f"   Duplicate patterns (skipped): {self.stats['duplicate_patterns']:,}")
        print(f"\n   Patterns per event:")
        for event, count in self.stats['patterns_per_event'].items():
            print(f"      {event}: {count:,}")
        print(f"\n   Sources:")
        for source, count in self.stats['sources'].items():
            print(f"      {source}: {count:,}")


def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Learn patterns from web for event detection')
    parser.add_argument('--target', type=int, default=1_000_000, 
                       help='Target number of patterns (default: 1,000,000)')
    parser.add_argument('--events', nargs='+', 
                       default=['explosion', 'fire', 'vehicle_accident', 'fighting', 'smoke'],
                       help='Events to learn patterns for')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for learned patterns')
    parser.add_argument('--merge', action='store_true',
                       help='Merge with existing patterns')
    
    args = parser.parse_args()
    
    # Create learner
    learner = WebPatternLearner(output_dir=args.output)
    
    # Load existing patterns if merge requested
    if args.merge:
        learner.load_existing_patterns()
    
    # Learn new patterns
    learner.learn_from_google(
        events=args.events,
        target_patterns=args.target,
        images_per_query=100
    )
    
    # Save patterns
    learner.save_patterns()
    
    print("\n✅ Pattern learning complete!")
    print(f"   Use these patterns by loading: {args.output}/web_learned_patterns_extended.pkl")


if __name__ == '__main__':
    main()
