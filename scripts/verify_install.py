"""Verify core dependencies are installed and usable.
Run: python scripts/verify_install.py
"""
from __future__ import annotations
import importlib, pkgutil, json

CORE_MODULES = [
    'flask', 'flask_socketio', 'sqlalchemy', 'ultralytics', 'cv2',
    'torch', 'tensorflow', 'numpy', 'pandas', 'sklearn', 'librosa',
]

results = {}
for m in CORE_MODULES:
    try:
        importlib.import_module(m)
        results[m] = 'OK'
    except Exception as e:
        results[m] = f'FAIL: {e.__class__.__name__}: {e}'

print(json.dumps(results, indent=2))

# Torch / CUDA quick info
try:
    import torch
    print('\nTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
except Exception:
    pass

try:
    import tensorflow as tf
    print('TensorFlow version:', tf.__version__)
    print('TF GPU devices:', tf.config.list_physical_devices('GPU'))
except Exception:
    pass
