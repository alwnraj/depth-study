"""Estimate relative depth from one image with a pinned ONNX checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import onnxruntime as ort

MODEL = Path(__file__).resolve().parent / 'models/depth-anything-v2-small/model.onnx'
REVISION = '34f40a4c93991327074907b51792dbc27e7d6f1c'
MODEL_SHA256 = 'afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c'
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def preprocess(image):
    width, height = image.size
    scale = 518 / max(width, height)
    # DPT's aspect-preserving resize rounds each dimension to a multiple of 14.
    resized = tuple(max(14, int(round(d * scale / 14)) * 14) for d in (width, height))
    pixels = np.asarray(image.resize(resized, Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    tensor = ((pixels - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    return tensor, resized


def visualization(raw, size):
    low, high = float(raw.min()), float(raw.max())
    normalized = np.zeros_like(raw) if high == low else (raw - low) / (high - low)
    # White is near, black is far; no inverse or metric conversion.
    image = Image.fromarray(np.uint8(np.clip(normalized * 255, 0, 255)))
    return image.resize(size, Image.Resampling.BILINEAR)


def run(image_path, output):
    image_path = Path(image_path).resolve(strict=True)
    output = Path(output)
    root = Path(__file__).resolve().parent
    if output.parent.resolve() != root or not output.name.startswith('results-'):
        raise ValueError('Output must be a results-* directory directly inside the repository.')
    output = output.resolve()
    if output.parent != root:
        raise ValueError('Output directory must not be a symlink outside the repository.')
    names = ('raw-output.npy', 'depth-visualization.png', 'report.json')
    if any((output / name).exists() for name in names):
        raise FileExistsError('Output files already exist; choose a new results-* directory.')
    if not MODEL.is_file():
        raise FileNotFoundError('Missing checkpoint. Run .venv/bin/python download_depth_model.py')
    if sha256(MODEL) != MODEL_SHA256:
        raise ValueError('Checkpoint SHA-256 does not match the pinned model.')
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    tensor, resized = preprocess(image)
    session = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
    raw_batch = session.run(['predicted_depth'], {'pixel_values': tensor})[0]
    if raw_batch.ndim != 3 or raw_batch.shape[0] != 1 or not np.isfinite(raw_batch).all():
        raise ValueError('Unexpected or non-finite model output.')
    raw = raw_batch[0]
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / names[0], raw, allow_pickle=False)
    visualization(raw, image.size).save(output / names[1])
    report = {
        'input': {'path': str(image_path), 'sha256': sha256(image_path), 'oriented_size_wh': list(image.size)},
        'model': {'name': 'onnx-community/depth-anything-v2-small', 'revision': REVISION,
                  'sha256': MODEL_SHA256, 'license': 'Apache-2.0', 'provider': 'CPUExecutionProvider'},
        'preprocessing': {'orientation': 'EXIF transpose', 'color': 'RGB', 'resize_wh': list(resized),
                          'resize': 'Pillow bicubic, aspect-preserving longest side 518, rounded to multiples of 14',
                          'normalization': '(RGB / 255 - mean) / std', 'mean': MEAN.tolist(), 'std': STD.tolist(),
                          'layout': 'NCHW float32'},
        'raw_output': {'file': names[0], 'shape_hw': list(raw.shape), 'dtype': str(raw.dtype),
                       'interpretation': 'relative inverse depth / disparity-like; larger means nearer; arbitrary per-image scale',
                       'min': float(raw.min()), 'max': float(raw.max())},
        'visualization': 'Per-image min-max grayscale; white near, black far; bilinear resize to oriented input size.',
        'limitations': ['Not metric distance or camera-space Z; scale is unknown.',
                        'Predictions may be wrong; hidden surfaces and physical object boundaries are not recovered.',
                        'Image depth has no GLB source-face mapping.']}
    (output / names[2]).write_text(json.dumps(report, indent=2) + '\n')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results-image-depth'))
    args = parser.parse_args()
    try:
        print(f'Wrote {run(args.image, args.output)}')
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
