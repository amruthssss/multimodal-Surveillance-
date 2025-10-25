"""Export action & (optionally) audio models to ONNX for CPU optimization.
Assumes the PyTorch .pth weights are compatible with the wrapper's model_factory placeholder.
Adjust model_factory to match your real architecture before exporting.
Usage:
  python scripts/export_onnx.py --model action --out models/action_model.onnx
  python scripts/export_onnx.py --model audio  --out models/audio_model.onnx
"""
from __future__ import annotations
import argparse
import torch
import os
from utils.action_wrapper import ActionWrapper
from utils.audio_wrapper import AudioWrapper


def export_action(out_path: str):
    wrapper = ActionWrapper(path='models/action_model.pth', device='cpu')
    dummy = torch.randn(1, 3, 224, 224)
    model = wrapper.model.eval()
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=['input'],
        output_names=['logits'],
        opset_version=17,
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
    )
    print(f"Exported action model to {out_path}")


def export_audio(out_path: str):
    wrapper = AudioWrapper(path='models/audio_model.pth', device='cpu')
    if wrapper.model is None:
        raise RuntimeError('Audio model not loaded or not TorchScript')
    dummy = torch.randn(1, 16000)  # adjust based on training input
    model = wrapper.model
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=['waveform'],
        output_names=['logits'],
        opset_version=17,
        dynamic_axes={'waveform': {0: 'batch'}},
    )
    print(f"Exported audio model to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['action','audio'], required=True)
    parser.add_argument('--out', required=True, help='Output ONNX file path')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.model == 'action':
        export_action(args.out)
    else:
        export_audio(args.out)

if __name__ == '__main__':
    main()
