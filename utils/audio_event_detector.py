"""YAMNet AudioSet event scores for short mono audio windows."""
from __future__ import annotations

import csv
import os
from math import gcd

import numpy as np
import onnxruntime as ort
from scipy.signal import resample_poly


class YamnetEventDetector:
    TARGET_RATE = 16_000

    def __init__(self, model_path: str, class_map_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        with open(class_map_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.class_indices = {
            row["display_name"].strip().lower(): int(row["index"])
            for row in rows
            if row.get("display_name") and row.get("index")
        }
        self.event_indices = {
            "gunshot": self._find_class("gunshot, gunfire"),
            "explosion": self._find_class("explosion"),
        }
        if any(index is None for index in self.event_indices.values()):
            raise ValueError("YAMNet class map is missing required event labels")

    def _find_class(self, name: str) -> int | None:
        return self.class_indices.get(name.lower())

    def predict(self, samples: np.ndarray, sample_rate: int) -> dict[str, float]:
        waveform = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not sample_rate or waveform.size == 0:
            return {"gunshot": 0.0, "explosion": 0.0}
        if sample_rate != self.TARGET_RATE:
            divisor = gcd(int(sample_rate), self.TARGET_RATE)
            waveform = resample_poly(
                waveform, self.TARGET_RATE // divisor, int(sample_rate) // divisor
            ).astype(np.float32, copy=False)
        waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        waveform = np.clip(waveform, -1.0, 1.0)
        if waveform.size < 15_360:
            waveform = np.pad(waveform, (0, 15_360 - waveform.size))
        scores = self.session.run(None, {"waveform": waveform})[0]
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim == 1:
            scores = scores[None, :]
        return {
            event: float(np.max(scores[:, index]))
            for event, index in self.event_indices.items()
        }
