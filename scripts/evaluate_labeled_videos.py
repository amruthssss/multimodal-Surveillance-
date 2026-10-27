"""Evaluate labeled video/audio clips against the live detection policy.

Manifest columns: media_path,label and optional media_type (video or audio).
Each clip should contain one primary label.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import Config
from enhanced_final_ultra_system import EnhancedUltraSystem
from utils.detection_policy import TemporalDetectionPolicy
from utils.feedback_learner import FeedbackThresholdLearner

LABELS = ('normal', 'explosion', 'fire', 'smoke', 'accident', 'gunshot')
CLASSES = LABELS[1:]


def evaluate_video(system, policy, path: Path, truth: str, sample_every: int) -> dict:
    system.reset_sequence_state()
    policy.reset('evaluation')
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f'Could not open video: {path}')
    predictions = []
    frame_index = 0
    sampled = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every == 0:
                raw = system.process_frame_for_api(frame, frame_index)
                if raw.get('event') == 'error':
                    raise RuntimeError(f'Frame {frame_index}: {raw.get("error", "inference failed")}')
                _, calibrated = policy.apply(raw, 'evaluation', max(1, frame_index))
                label = calibrated['label']
                if label != 'normal':
                    predictions.append((label, float(calibrated['confidence'])))
                sampled += 1
            frame_index += 1
    finally:
        capture.release()

    if not sampled:
        raise RuntimeError(f'Video contained no readable sampled frames: {path}')
    predicted = 'normal'
    if predictions:
        # Prefer the most persistent event, then the strongest mean confidence.
        scores = {}
        for label, confidence in predictions:
            scores.setdefault(label, []).append(confidence)
        predicted = max(scores, key=lambda label: (len(scores[label]), sum(scores[label]) / len(scores[label])))
    return {
        'media': str(path), 'mediaType': 'video', 'actual': truth, 'predicted': predicted,
        'sampledFrames': sampled, 'eventFrames': len(predictions),
    }


def evaluate_audio(system, path: Path, truth: str) -> dict:
    if system.audio_event_model is None:
        raise RuntimeError('YAMNet audio model is unavailable; check its configured local model weights.')
    waveform, sample_rate = sf.read(path, dtype='float32', always_2d=True)
    waveform = waveform.mean(axis=1)
    streaks = {'gunshot': 0, 'explosion': 0}
    detected = []
    peak_scores = {'gunshot': 0.0, 'explosion': 0.0}
    for start in range(0, len(waveform), int(sample_rate)):
        window = waveform[start:start + int(sample_rate)]
        if not window.size:
            continue
        scores = system.classify_audio_events(window, int(sample_rate))
        for label in streaks:
            peak_scores[label] = max(peak_scores[label], float(scores.get(label, 0.0)))
            threshold = system.audio_event_thresholds.get(label, 0.2)
            streaks[label] = streaks[label] + 1 if scores.get(label, 0.0) >= threshold else 0
            if streaks[label] >= 2:
                detected.append(label)
    candidates = peak_scores
    predicted = max(detected, key=lambda label: candidates[label]) if detected else 'normal'
    return {
        'media': str(path), 'mediaType': 'audio', 'actual': truth,
        'predicted': predicted, 'scores': candidates,
        'triggeredWindows': len(detected),
    }


def summarize(rows: list[dict]) -> dict:
    metrics = {}
    for label in CLASSES:
        tp = sum(row['actual'] == label and row['predicted'] == label for row in rows)
        fp = sum(row['actual'] != label and row['predicted'] == label for row in rows)
        fn = sum(row['actual'] == label and row['predicted'] != label for row in rows)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        metrics[label] = {
            'truePositives': tp, 'falsePositives': fp, 'falseNegatives': fn,
            'precision': precision, 'recall': recall, 'f1': f1,
        }
    return {
        'clips': len(rows),
        'accuracy': sum(row['actual'] == row['predicted'] for row in rows) / len(rows) if rows else None,
        'metrics': metrics,
        'results': rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path, help='CSV with media_path,label and optional media_type columns')
    parser.add_argument('--sample-every', type=int, default=6, help='Analyze every Nth frame (default: 6)')
    parser.add_argument('--output', type=Path, help='Optional path for a JSON metrics report')
    args = parser.parse_args()
    if args.sample_every < 1:
        parser.error('--sample-every must be at least 1')
    if not args.manifest.is_file():
        parser.error(f'Manifest not found: {args.manifest}')

    with args.manifest.open(newline='', encoding='utf-8-sig') as source:
        manifest = list(csv.DictReader(source))
    if not manifest or 'label' not in manifest[0] or not ({'media_path'} <= set(manifest[0]) or {'video_path'} <= set(manifest[0])):
        parser.error('Manifest must contain media_path,label (or legacy video_path,label) columns and at least one row')
    clips = []
    for row in manifest:
        path = Path(row.get('media_path') or row.get('video_path') or '').expanduser()
        if not path.is_absolute():
            path = (args.manifest.resolve().parent / path).resolve()
        label = row['label'].strip().lower()
        if label not in LABELS:
            parser.error(f'Unsupported label {label!r}; choose from {", ".join(LABELS)}')
        media_type = (row.get('media_type') or 'video').strip().lower()
        if media_type not in {'video', 'audio'}:
            parser.error(f'Unsupported media_type {media_type!r}; choose video or audio')
        if media_type == 'video' and label == 'gunshot':
            parser.error(f'{path}: gunshot detection uses audio; mark the clip as audio.')
        if media_type == 'audio' and label not in {'normal', 'gunshot', 'explosion'}:
            parser.error(f'{path}: audio evaluation supports normal, gunshot, and explosion labels.')
        if not path.is_file():
            parser.error(f'Media file not found: {path}')
        clips.append((path, label, media_type))

    learner = FeedbackThresholdLearner(str(Config.DATA_DIR / 'detection_feedback.json'))
    policy = TemporalDetectionPolicy(learner)
    system = EnhancedUltraSystem(use_audio=False)
    results = []
    for index, (path, label, media_type) in enumerate(clips, 1):
        print(f'[{index}/{len(clips)}] {path.name} ({media_type}: {label})', flush=True)
        if media_type == 'audio':
            results.append(evaluate_audio(system, path, label))
        else:
            results.append(evaluate_video(system, policy, path, label, args.sample_every))
    report = summarize(results)
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Report saved: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
