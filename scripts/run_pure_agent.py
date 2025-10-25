"""
Run Pure Pattern-Based Agent (NO YOLO)
Generates test_PURE_AGENT.mp4 using pattern-only detection from `pure_agent_enhanced.py` if available,
otherwise uses `EnhancedPureAgent` code present in repo.

Usage:
    python scripts/run_pure_agent.py --input test.mp4 --output test_PURE_AGENT.mp4
"""

import argparse
from pathlib import Path
import cv2

# Try to import EnhancedPureAgent from pure_agent_enhanced.py
try:
    from pure_agent_enhanced import EnhancedPureAgent
    AGENT_CLASS = EnhancedPureAgent
except Exception:
    # Try fallback to HybridIntelligentAgent pattern matching from hybrid_intelligent_agent
    try:
        from hybrid_intelligent_agent import HybridIntelligentAgent
        class FallbackPureAgent:
            def __init__(self):
                print('Using HybridIntelligentAgent for pattern-only detection (no YOLO)')
                self.hybrid = HybridIntelligentAgent()
            def detect(self, frame):
                # hybrid.detect expects a frame and returns dict
                return self.hybrid.detect(frame)
        AGENT_CLASS = FallbackPureAgent
    except Exception as e:
        print('No suitable pure agent found in repository:', e)
        AGENT_CLASS = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='test.mp4')
    p.add_argument('--output', default='test_PURE_AGENT.mp4')
    return p.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    if AGENT_CLASS is None:
        print('No agent available. Aborting.')
        return

    agent = AGENT_CLASS()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f'Failed to open {input_path}')
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    import time
    start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detection = None
        try:
            detection = agent.detect(frame)
        except Exception as e:
            # If agent.detect signature differs, try alternative
            try:
                detection = agent.hybrid.detect(frame)
            except:
                detection = None

        # Draw detection if present
        if detection and detection.get('event_type'):
            label = f"{detection['event_type']}: {detection.get('confidence',0):.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - start
            fps_proc = frame_idx / (elapsed + 1e-6)
            print(f"Processed: {frame_idx}/{total} | {fps_proc:.2f} FPS")

    cap.release()
    out.release()
    print(f"Output saved: {output_path}")

if __name__ == '__main__':
    main()
