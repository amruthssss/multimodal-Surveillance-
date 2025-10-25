"""Action recognition wrapper (PyTorch EfficientNet placeholder).
Provide a model_factory callable returning an uninitialized model instance.
"""
from __future__ import annotations
from typing import Tuple, Callable
import torch
import torchvision.transforms as T
from PIL import Image
import cv2

class ActionWrapper:
    def __init__(self, path: str = 'models/action_model.pth', device: str = 'cpu', model_factory: Callable | None = None):
        self.device = device
        if model_factory is None:
            # minimal example: small CNN placeholder
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
        self.model = model_factory()
        try:
            state = torch.load(path, map_location=device)
            if isinstance(state, dict) and 'state_dict' in state:
                self.model.load_state_dict(state['state_dict'], strict=False)
            else:
                self.model.load_state_dict(state, strict=False)
        except Exception:
            # leave random weights for placeholder
            pass
        self.model.to(device).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.labels = ['fighting', 'collapsing', 'theft', 'fire', 'normal']

    def predict(self, frame) -> Tuple[str, float]:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).cpu())
            return self.labels[idx], float(probs[idx].cpu())
