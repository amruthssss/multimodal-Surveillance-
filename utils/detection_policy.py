"""Shared temporal alert policy for live inference and offline evaluation."""

from __future__ import annotations

import time
from collections import deque


class TemporalDetectionPolicy:
    EVENTS = {'EXPLOSION', 'FIRE', 'ACCIDENT', 'SMOKE', 'GUNSHOT'}

    def __init__(self, learner):
        self.learner = learner
        self.history = {}

    def reset(self, camera_id=None):
        if camera_id is None:
            self.history.clear()
        else:
            self.history.pop(str(camera_id), None)

    def apply(self, raw_result, camera_id, frame_num=0):
        result = dict(raw_result)
        source_event = result.get('candidate_event') if frame_num > 0 else result.get('event')
        source_confidence = result.get('candidate_confidence') if frame_num > 0 else result.get('confidence')
        event = str(source_event or 'normal').upper()
        confidence = max(0.0, min(1.0, float(source_confidence or 0.0) / 100.0))
        event = event if event in self.EVENTS else 'NORMAL'
        confirmed = event != 'NORMAL'
        count = 1 if confirmed else 0

        if frame_num > 0:
            now = time.monotonic()
            history = self.history.setdefault(str(camera_id), deque(maxlen=5))
            while history and now - history[0][0] > 3.0:
                history.popleft()
            minimum = self.learner.threshold_for(event.lower()) if confirmed else 1.0
            passed = confirmed and confidence >= minimum
            history.append((now, event if passed else 'NORMAL', confidence if passed else 0.0))
            recent = list(history)[-3:]

            if confirmed:
                count = sum(item[1] == event for item in recent)
            else:
                consensus = max(
                    (name for name in {item[1] for item in recent} if name != 'NORMAL'),
                    key=lambda name: sum(item[1] == name for item in recent),
                    default='NORMAL',
                )
                count = sum(item[1] == consensus for item in recent)
                if consensus != 'NORMAL' and count >= 2:
                    event = consensus
                    confidence = sum(item[2] for item in recent if item[1] == consensus) / count
                    confirmed = True
                else:
                    count = 0
            confirmed = confirmed and count >= 2

        reasoning = result.get('reasoning', {})
        critical = confirmed and count >= 3 and confidence >= 0.85
        if confirmed and event == 'EXPLOSION' and count >= 2 and confidence >= 0.95:
            detail = str(reasoning.get('explosion', '')).lower() if isinstance(reasoning, dict) else str(reasoning).lower()
            cues = ('flash', 'smoke', 'blast', 'debris', 'motion spike')
            critical = sum(cue in detail for cue in cues) >= 3
        risk = 'CRITICAL' if critical else 'HIGH' if confirmed else 'LOW'
        label = event.lower() if confirmed else 'normal'
        result.update({
            'event': label,
            'risk_level': risk,
            'confidence': confidence * 100.0 if confirmed else 0.0,
            'candidate_event': event.lower(),
            'candidate_confidence': confidence * 100.0,
            'confirmation_count': count,
            'should_alert': confirmed,
        })
        final = {
            'label': label,
            'risk': risk,
            'confidence': confidence if confirmed else 0.0,
            'candidate_confidence': confidence,
            'candidate_event': event.lower(),
            'confirmation_count': count,
            'reasoning': reasoning,
            'should_alert': confirmed,
            'source': 'enhanced_visual_agent',
        }
        return result, final
