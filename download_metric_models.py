"""Download pinned metric-depth ONNX exports and verify SHA-256 hashes."""
import os
from pathlib import Path
from urllib.request import urlopen

from image_depth import sha256

ROOT = Path(__file__).resolve().parent / 'models' / 'depth-anything-v2-metric'
MODELS = {
    'indoor': {
        'revision': 'a4259a3c45137b6eb32c84fcd95b86cd54c255b9',
        'repo': '77ukhtar/depth-anything-v2-metric-onnx',
        'files': {'model.onnx': 'badcaa28c923da4b0bfaa370ed709acfa00e9f743d295d5443e2149a383413c9'},
    },
    'outdoor': {
        'revision': 'f3f42bbfb266b8c7dbd0ef6cf24bfc4b167717f5',
        'repo': 'sollaholla/depth-anything-v2-metric-vits-vkitti',
        'files': {
            'depth_anything_v2_metric_vits_vkitti_518x518.onnx': 'd2962720aa4c73677bff72f230f814df80d60b7d72759cf1d36f3846c979b568',
            'depth_anything_v2_metric_vits_vkitti_518x518.onnx.data': 'eb38b775e6247f59dbef340364cf6c991b87cbf221a92937018777ec51a84431',
        },
    },
}


def main():
    for mode, model in MODELS.items():
        directory = ROOT / mode
        directory.mkdir(parents=True, exist_ok=True)
        for name, expected in model['files'].items():
            target = directory / name
            if target.is_file() and sha256(target) == expected:
                continue
            temporary = target.with_suffix(target.suffix + '.download')
            url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{name}"
            print(f'Downloading {mode}: {name}', flush=True)
            try:
                with urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                if sha256(temporary) != expected:
                    raise ValueError(f'Checksum mismatch: {name}')
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        print(f'Verified {mode} model in {directory}', flush=True)


if __name__ == '__main__':
    main()
