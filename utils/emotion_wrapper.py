"""Emotion recognition wrapper for a Keras .h5 model."""
from __future__ import annotations
import cv2
import numpy as np
import os
import json
from typing import Tuple

try:
    from tensorflow.keras.models import load_model, model_from_json  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    load_model = None  # type: ignore
    model_from_json = None  # type: ignore
    Adam = None  # type: ignore
    tf = None  # type: ignore

DEFAULT_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionWrapper:
    def __init__(self, path: str = 'models/emotion_model.h5', labels=None):
        self.labels = labels or DEFAULT_LABELS
        self.model = None
        self.error = None
        self.model_path = path
        
        if load_model:
            # Try multiple loading strategies
            success = (
                self._try_load_h5(path) or 
                self._try_load_savedmodel(path) or 
                self._try_load_from_parts(path) or
                self._try_load_compatible_h5(path)
            )
            
            if success:
                print(f"Successfully loaded emotion model from {path}")
            else:
                print(f"All loading attempts failed for emotion model: {self.error}")
                
    def _try_load_h5(self, path: str) -> bool:
        """Try loading original H5 file"""
        try:
            self.model = load_model(path)
            return True
        except Exception as e:
            self.error = f"H5 loading failed: {e}"
            return False
            
    def _try_load_savedmodel(self, path: str) -> bool:
        """Try loading SavedModel format"""
        try:
            savedmodel_path = path.replace('.h5', '_savedmodel')
            if os.path.exists(savedmodel_path):
                self.model = tf.keras.models.load_model(savedmodel_path)
                return True
        except Exception as e:
            self.error = f"SavedModel loading failed: {e}"
        return False
        
    def _try_load_from_parts(self, path: str) -> bool:
        """Try loading from separate architecture and weights"""
        try:
            base_path = path.replace('.h5', '')
            arch_path = f"{base_path}_architecture.json"
            weights_path = f"{base_path}.weights.h5"  # Updated for new format
            
            if os.path.exists(arch_path) and os.path.exists(weights_path):
                # Load architecture
                with open(arch_path, 'r') as f:
                    model_json = f.read()
                self.model = model_from_json(model_json)
                
                # Load weights
                self.model.load_weights(weights_path)
                
                # Recompile
                if Adam:
                    opt = Adam(learning_rate=0.0001)
                    self.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
                
                return True
        except Exception as e:
            self.error = f"Architecture+Weights loading failed: {e}"
        return False
        
    def _try_load_compatible_h5(self, path: str) -> bool:
        """Try loading compatible H5 version"""
        try:
            compatible_path = path.replace('.h5', '_compatible.h5')
            if os.path.exists(compatible_path):
                self.model = load_model(compatible_path)
                return True
        except Exception as e:
            self.error = f"Compatible H5 loading failed: {e}"
        return False

    def predict_from_face(self, face_bgr) -> Tuple[str, float]:
        if self.model is None or face_bgr is None:
            return ('Neutral', 0.0)
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (48, 48))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # (1,48,48,1)
        try:
            preds = self.model.predict(img, verbose=0)[0]
            idx = int(np.argmax(preds))
            return self.labels[idx], float(preds[idx])
        except Exception:
            return ('Neutral', 0.0)
