"""Run the local single-photo depth demo on localhost."""
import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import onnxruntime as ort

from image_depth import MODEL, MODEL_SHA256, MEAN, STD, preprocess, sha256, visualization
from download_metric_models import ROOT as METRIC_ROOT, MODELS

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 20 * 1024 * 1024


def encode_png(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode('ascii')


def estimate(data, session, metric_session=None, scene="outdoor"):
    with Image.open(io.BytesIO(data)) as original:
        if original.width * original.height > 30_000_000:
            raise ValueError('Image is too large; use an image under 30 megapixels.')
        image = ImageOps.exif_transpose(original).convert('RGB')
    tensor, _ = preprocess(image)
    raw = session.run(['predicted_depth'], {'pixel_values': tensor})[0][0]
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('The model returned an invalid depth map.')
    metric_values = None
    if metric_session is not None:
        pixels = np.asarray(image.resize((518, 518), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
        metric_tensor = ((pixels - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        metric_raw = metric_session.run(None, {metric_session.get_inputs()[0].name: metric_tensor})[0]
        metric_raw = np.squeeze(metric_raw)
        if metric_raw.ndim != 2 or not np.isfinite(metric_raw).all() or np.any(metric_raw < 0):
            raise ValueError('The metric model returned an invalid depth map.')
        metric_values = np.asarray(Image.fromarray(metric_raw).resize((256, 256), Image.Resampling.BILINEAR))
    width = 160
    height = max(2, round(width * image.height / image.width))
    if height > 400:
        height = 400
        width = max(2, round(height * image.width / image.height))
    sampled = np.asarray(Image.fromarray(raw).resize((width, height), Image.Resampling.BILINEAR))
    lo, hi = np.percentile(sampled, [2, 98])
    near = np.zeros_like(sampled) if hi == lo else np.clip((sampled - lo) / (hi - lo), 0, 1)
    color = image.resize((width, height), Image.Resampling.LANCZOS)
    # Limit display size; original input is unchanged and never written by the server.
    display = image.copy()
    display.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    depth = visualization(raw, display.size)
    return {'image': encode_png(display), 'depth': encode_png(depth),
            'colors': encode_png(color), 'width': width, 'height': height,
            'near': near.reshape(-1).round(4).tolist(), 'original_size': list(image.size),
            'metric': None if metric_values is None else {'values': metric_values.reshape(-1).round(3).tolist(),
                'width': 256, 'height': 256, 'scene': scene}}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/app.js': ('app.js', 'application/javascript'), '/style.css': ('style.css', 'text/css')}
        if self.path not in files:
            self.send_error(404)
            return
        name, content_type = files[self.path]
        content = (ROOT / 'depth_demo' / name).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path != '/estimate':
            self.send_error(404)
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if size <= 0 or size > MAX_UPLOAD:
                raise ValueError('Choose an image smaller than 20 MB.')
            data = self.rfile.read(size)
            scene = self.headers.get('X-Scene', 'outdoor')
            if scene not in ('indoor', 'outdoor'):
                raise ValueError('Choose indoor or outdoor.')
            result = estimate(data, self.server.session, self.server.metric_sessions.get(scene), scene)
            payload = json.dumps(result, allow_nan=False).encode()
            status = 200
        except (ValueError, OSError, UnidentifiedImageError) as exc:
            payload = json.dumps({'error': str(exc) or 'Could not read that image.'}).encode()
            status = 400
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if not MODEL.is_file() or sha256(MODEL) != MODEL_SHA256:
        raise SystemExit('Missing or incorrect model. Run .venv/bin/python download_depth_model.py')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.session = ort.InferenceSession(str(MODEL), providers=['CPUExecutionProvider'])
    server.metric_sessions = {}
    for scene, config in MODELS.items():
        directory = METRIC_ROOT / scene
        if all((directory / name).is_file() and sha256(directory / name) == digest for name, digest in config['files'].items()):
            graph = next(name for name in config['files'] if name.endswith('.onnx'))
            server.metric_sessions[scene] = ort.InferenceSession(str(directory / graph), providers=['CPUExecutionProvider'])
        else:
            print(f'{scene} metric model missing or incorrect; run .venv/bin/python download_metric_models.py', flush=True)
    print(f'Depth demo: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
