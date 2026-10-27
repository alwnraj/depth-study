"""Preview and benchmark single-frame depth on a local camera. Press Q to stop."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from image_depth import MEAN, MODEL, MODEL_SHA256, STD, preprocess, sha256
from download_metric_models import MODELS, ROOT as METRIC_ROOT


def load_model(mode):
    if mode == 'relative':
        if not MODEL.is_file() or sha256(MODEL) != MODEL_SHA256:
            raise ValueError('Relative model missing or invalid. Run download_depth_model.py.')
        model = MODEL
    else:
        spec = MODELS[mode]
        directory = METRIC_ROOT / mode
        if any(not (directory / name).is_file() or sha256(directory / name) != digest
               for name, digest in spec['files'].items()):
            raise ValueError('Metric model missing or invalid. Run download_metric_models.py.')
        model = directory / next(name for name in spec['files'] if name.endswith('.onnx'))
    session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
    return session, session.get_inputs()[0].name


def model_input(frame, mode):
    rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if mode == 'relative':
        return preprocess(rgb)[0]
    pixels = np.asarray(rgb.resize((518, 518), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return ((pixels - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def depth_display(depth, mode, size):
    depth = np.squeeze(depth)
    if depth.ndim != 2 or not np.isfinite(depth).all():
        raise ValueError('Model returned an invalid depth map.')
    lo, hi = float(depth.min()), float(depth.max())
    normalized = np.zeros_like(depth) if hi == lo else (depth - lo) / (hi - lo)
    if mode != 'relative':
        normalized = 1 - normalized  # Metric depth increases with distance; white remains near.
    grayscale = np.uint8(np.clip(normalized * 255, 0, 255))
    return cv2.resize(grayscale, size, interpolation=cv2.INTER_LINEAR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['relative', 'indoor', 'outdoor'], default='outdoor')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=15,
                        help='Timed benchmark after two warm-up frames; 0 runs until Q (default: 15).')
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error('--seconds must be nonnegative')
    try:
        session, input_name = load_model(args.mode)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        parser.exit(1, f'Could not open camera {args.camera}. Check the device and camera permission.\n')
    count, inference_total, started, elapsed = 0, 0.0, None, 0.0
    print(f'Camera {args.camera}; {args.mode} model; CPU. Press Q to stop.', flush=True)
    try:
        deadline = time.monotonic() + 8
        for index in range(2):
            ok, frame = cap.read()
            while not ok and time.monotonic() < deadline:
                time.sleep(0.1)
                ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera opened but returned no frames during warm-up.')
            session.run(None, {input_name: model_input(frame, args.mode)})
        started = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera stopped returning frames.')
            tensor = model_input(frame, args.mode)
            before = time.perf_counter()
            raw = session.run(None, {input_name: tensor})[0]
            inference_total += time.perf_counter() - before
            gray = depth_display(raw, args.mode, (frame.shape[1], frame.shape[0]))
            view = np.concatenate((frame, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)), axis=1)
            count += 1
            elapsed = time.perf_counter() - started
            cv2.putText(view, f'{args.mode} | {count / elapsed:.1f} FPS | white = near | Q to quit',
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 210, 255), 2, cv2.LINE_AA)
            cv2.imshow('Live depth - camera / depth', view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or (args.seconds and time.perf_counter() - started >= args.seconds):
                break
    except (RuntimeError, ValueError, cv2.error) as exc:
        print(f'Error: {exc}', flush=True)
    finally:
        if started is not None:
            elapsed = time.perf_counter() - started
        cap.release()
        cv2.destroyAllWindows()
    if count:
        print(f'Processed {count} frames in {elapsed:.2f} s: {count / elapsed:.2f} end-to-end FPS; '
              f'mean inference {1000 * inference_total / count:.1f} ms/frame '
              f'({count / inference_total:.2f} inference FPS).', flush=True)
        print('End-to-end includes capture, preprocessing, inference and display; camera frames are not saved.', flush=True)
    else:
        parser.exit(1, 'No benchmark frames were processed.\n')


if __name__ == '__main__':
    main()
