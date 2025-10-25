"""Recorder buffers frames per camera and saves short clips when requested."""
from __future__ import annotations
import os
import time
import cv2
from collections import deque
from typing import Dict, Deque, List

class Recorder:
    def __init__(self, save_root: str = 'data/uploads', fps: int = 10, seconds: int = 5):
        self.save_root = save_root
        os.makedirs(save_root, exist_ok=True)
        self.buffer_max = fps * seconds
        self.buffers: Dict[int, Deque] = {}
        self.fps = fps

    def add_frame(self, cam_id: int, frame):
        if cam_id not in self.buffers:
            self.buffers[cam_id] = deque(maxlen=self.buffer_max)
        self.buffers[cam_id].append(frame.copy())

    def save_clip(self, cam_id: int, prefix: str = 'event') -> str | None:
        if cam_id not in self.buffers or not self.buffers[cam_id]:
            return None
        t = int(time.time())
        out_dir = os.path.join(self.save_root, str(cam_id))
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, f"{prefix}_{t}.mp4")
        frames: List = list(self.buffers[cam_id])
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        return filepath
