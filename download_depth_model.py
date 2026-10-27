"""Download and verify the pinned Depth Anything V2 Small ONNX checkpoint."""
import os
from pathlib import Path
from urllib.request import urlopen

from image_depth import MODEL, MODEL_SHA256, REVISION, sha256

URL = ('https://huggingface.co/onnx-community/depth-anything-v2-small/'
       f'resolve/{REVISION}/onnx/model.onnx')


def main():
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    if MODEL.is_file() and sha256(MODEL) == MODEL_SHA256:
        print(f'Verified {MODEL}')
        return
    temporary = MODEL.with_suffix('.download')
    try:
        with urlopen(URL, timeout=120) as response, temporary.open('wb') as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        if sha256(temporary) != MODEL_SHA256:
            raise ValueError('Downloaded checkpoint checksum mismatch')
        os.replace(temporary, MODEL)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Verified {MODEL}')


if __name__ == '__main__':
    main()
