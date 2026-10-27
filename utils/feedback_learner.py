"""Human-rewarded online threshold tuning for live event detections.

This deliberately learns operating thresholds, not neural-network weights. It
needs explicit feedback because unlabeled video cannot tell a true event from a
false alarm reliably.
"""

from __future__ import annotations

import json
import os
import threading


class FeedbackThresholdLearner:
    BASE_THRESHOLDS = {
        'explosion': 0.70,
        'fire': 0.70,
        'accident': 0.72,
        'smoke': 0.72,
        'gunshot': 0.75,
    }

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._state = {'thresholds': {}, 'feedback': {'correct': 0, 'false_alarm': 0, 'missed': 0}}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, 'r', encoding='utf-8') as source:
                loaded = json.load(source)
            if isinstance(loaded, dict):
                self._state['thresholds'] = loaded.get('thresholds', {})
                self._state['feedback'] = loaded.get('feedback', self._state['feedback'])
        except (OSError, ValueError, TypeError):
            pass

    def threshold_for(self, event: str) -> float:
        event = str(event).lower()
        base = self.BASE_THRESHOLDS.get(event, 0.70)
        try:
            learned = float(self._state['thresholds'].get(event, base))
        except (TypeError, ValueError):
            learned = base
        return min(0.95, max(0.55, learned))

    def record(self, event: str, outcome: str) -> dict:
        event = str(event).lower()
        if event not in self.BASE_THRESHOLDS:
            raise ValueError('Unsupported event label')
        if outcome not in {'correct', 'false_alarm', 'missed'}:
            raise ValueError('Unsupported feedback outcome')
        with self._lock:
            threshold = self.threshold_for(event)
            # Reward correct detections with a small recall adjustment; penalize
            # false alarms to favor precision and lower the cutoff after a miss.
            threshold += {'correct': -0.005, 'false_alarm': 0.025, 'missed': -0.020}[outcome]
            threshold = min(0.95, max(0.55, threshold))
            self._state['thresholds'][event] = round(threshold, 4)
            self._state['feedback'][outcome] = int(self._state['feedback'].get(outcome, 0)) + 1
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            temporary = self.path + '.tmp'
            with open(temporary, 'w', encoding='utf-8') as target:
                json.dump(self._state, target, indent=2)
            os.replace(temporary, self.path)
            return {'event': event, 'threshold': threshold, 'outcome': outcome, 'feedback': dict(self._state['feedback'])}
