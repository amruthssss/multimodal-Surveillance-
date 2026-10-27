"""Action recognition wrapper for a locally trained PyTorch checkpoint."""
from __future__ import annotations
from typing import Tuple, Callable
import os
import torch
import torchvision.transforms as T
from PIL import Image
import cv2

class ActionWrapper:
    def __init__(self, path: str = 'models/action_model.pth', device: str = 'cpu', model_factory: Callable | None = None):
        self.device = device
        self.model = None
        self.error = None
        self.path = path
        self.labels = ['fighting', 'collapsing', 'theft', 'fire', 'normal']
        if not os.path.isfile(path):
            self.error = f'Action checkpoint not found: {path}'
            return

        if model_factory is None:
            # This architecture is only a checkpoint loader scaffold. It is never
            # used for inference unless a compatible trained checkpoint loads.
            import torch.nn as nn
            class TinyNet(nn.Module):
                def __init__(self, num_classes=5):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes)
                    )
                def forward(self, x):
                    return self.net(x)
            model_factory = lambda: TinyNet()
        try:
            candidate = model_factory()
            state = torch.load(path, map_location=device, weights_only=True)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            if not isinstance(state, dict):
                raise ValueError('Checkpoint must contain a state dictionary')
            candidate.load_state_dict(state, strict=True)
            self.model = candidate.to(device).eval()
        except Exception as error:
            self.error = f'Could not load action checkpoint: {error}'
            self.model = None
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, frame) -> Tuple[str, float]:
        if self.model is None or frame is None:
            return ('unavailable', 0.0)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).cpu())
            return self.labels[idx], float(probs[idx].cpu())

    def classify(self, frame):
        """Return the legacy agent result shape when a trained model is loaded."""
        if self.model is None:
            return None
        label, confidence = self.predict(frame)
        return {'action': label.title(), 'confidence': confidence}
