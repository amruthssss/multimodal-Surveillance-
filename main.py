"""
MAIN FILE - Multi-Modal Surveillance System
Detects: Accidents, Explosions, Fire, Smoke, Fighting
Uses 221,660+ learned patterns from web sources

Usage:
    python main.py                           # Interactive mode
    python main.py --video "path/to/video.mp4"
    python main.py --video "D:/Videos/test.mp4" --save-results
"""

import argparse
import os
import cv2
import time
import json
from datetime import datetime
from enhanced_final_ultra_system import EnhancedUltraSystem

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Local Video Analysis with Enhanced AI')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--save-results', action='store_true', help='Save JSON results')
    parser.add_argument('--no-display', action='store_true', help='Run without display')
    parser.add_argument('--output-dir', type=str, default='output/', help='Output directory')
    return parser.parse_args()

def format_time(seconds):
    """Convert seconds to HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    """Main entry point for the surveillance system"""
    
    args = parse_args()
    
    print("=" * 80)
    print("🚨 MULTI-MODAL SURVEILLANCE SYSTEM")
    print("=" * 80)
    print("Detects:")
    print("  🚗 Vehicle Accidents")
    print("  💥 Explosions")
    print("  🔥 Fire")
    print("  💨 Smoke")
    print("  👊 Fighting")
    print()
    print("Using 221,660+ learned patterns (80% Agent + 20% YOLO)")
    print("=" * 80)
    print()
    
    # Get video path
    if args.video:
        video_path = args.video.strip('"').strip("'")
    else:
        video_path = input("Enter video path (or press Enter for camera 0): ").strip()
        if not video_path:
            print("\nUsing default camera (0)...")
            video_path = 0
        else:
            video_path = video_path.strip('"').strip("'")
    
    # Validate video file
    if isinstance(video_path, str) and not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Generate output paths
    if isinstance(video_path, str):
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(video_dir or '.', f"{video_name}_detected.mp4")
        results_path = os.path.join(args.output_dir, f"{video_name}_results.json")
    else:
        output_path = os.path.join(args.output_dir, "camera_detected.mp4")
        results_path = os.path.join(args.output_dir, "camera_results.json")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n📹 Input:  {video_path}")
    print(f"💾 Output: {output_path}")
    if args.save_results:
        print(f"📄 Results: {results_path}")
    print()
    
    # Create system (without audio for faster processing)
    print("🔧 Initializing Enhanced Ultra Hybrid System...")
    print("   Loading YOLOv11m model...")
    print("   Loading 221,660 learned patterns...")
    system = EnhancedUltraSystem(
        model_path='runs/detect/train/weights/best.pt',
        use_audio=False
    )
    
    print()
    print("✅ System Ready!")
    print()
    print("▶️  Processing video...")
    print("   (Press 'q' in video window to stop)")
    print()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video: {width}x{height} @ {fps}fps | Total: {total_frames} frames")
    print()
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Detection storage
    detections = []
    alerts = []
    
    frame_idx = 0
    start_time = time.time()
    
    print("="*80)
    print(f"{'Frame':<10} {'Time':<12} {'Event':<15} {'Risk':<8} {'Confidence':<12}")
    print("="*80)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            current_time = frame_idx / fps
            
            # Process frame
            result = system.process_frame_for_api(frame, frame_idx)
            
            event = result.get('event', 'normal')
            confidence = result.get('confidence', 0.0)
            risk = result.get('risk_level', 'LOW')
            
            # Store significant detections
            if confidence > 0.5 and event != 'normal':
                detection = {
                    'frame': frame_idx,
                    'time': format_time(current_time),
                    'event': event,
                    'confidence': round(confidence, 3),
                    'risk': risk,
                    'reasoning': result.get('reasoning', '')
                }
                detections.append(detection)
                
                print(f"{frame_idx:<10} {format_time(current_time):<12} "
                      f"{event:<15} {risk:<8} {confidence*100:>5.1f}%")
                
                if risk == 'HIGH':
                    alerts.append(detection)
                
                # Annotate frame
                cv2.rectangle(frame, (10, 10), (width-10, 100), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (width-10, 100), (0, 255, 255), 2)
                cv2.putText(frame, f"Event: {event.upper()}", (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                risk_color = (0, 0, 255) if risk == 'HIGH' else (0, 165, 255) if risk == 'MEDIUM' else (0, 255, 0)
                cv2.putText(frame, f"Risk: {risk}", (20, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
                cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (20, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Display
            if not args.no_display:
                display = cv2.resize(frame, (1280, 720)) if width > 1280 else frame
                cv2.imshow('Video Analysis', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_idx % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                elapsed = time.time() - start_time
                current_fps = frame_idx / elapsed
                print(f"📊 Progress: {progress:.1f}% | FPS: {current_fps:.1f} | "
                      f"Detections: {len(detections)}")
    
    except KeyboardInterrupt:
        print("\n⏸️  Stopped by user")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Statistics
    elapsed = time.time() - start_time
    
    print()
    print("="*80)
    print("✅ PROCESSING COMPLETE!")
    print("="*80)
    print(f"⏱️  Time: {format_time(elapsed)}")
    print(f"🎬 Frames: {frame_idx} / {total_frames}")
    print(f"⚡ FPS: {frame_idx/elapsed:.2f}")
    print(f"🎯 Detections: {len(detections)}")
    print(f"⚠️  Alerts: {len(alerts)}")
    print(f"💾 Saved: {output_path}")
    
    # Save JSON results
    if args.save_results and detections:
        results = {
            'video': video_path,
            'processing_time': format_time(elapsed),
            'total_detections': len(detections),
            'high_risk_alerts': len(alerts),
            'detections': detections,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Results: {results_path}")
    
    # Show summary
    if detections:
        print(f"\n📋 Event Summary:")
        events = {}
        for d in detections:
            events[d['event']] = events.get(d['event'], 0) + 1
        for event, count in events.items():
            print(f"   {event.capitalize()}: {count}")
    
    if alerts:
        print(f"\n⚠️  High Risk Alerts:")
        for alert in alerts[:5]:
            print(f"   {alert['time']} - {alert['event'].upper()} ({alert['confidence']*100:.1f}%)")
    
    print("\n" + "="*80)
    print("✅ Done! Check output folder for results.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
