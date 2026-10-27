import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from render_views import render_view, render_views
from segment_objects import write_glb


def _textured_glb(path):
    texture_path = Path(path).with_suffix(".png")
    texture_pixels = np.array([[[230, 25, 15], [20, 220, 30]],
                               [[15, 40, 230], [220, 210, 10]]], dtype=np.uint8)
    Image.fromarray(texture_pixels).save(texture_path)
    png_bytes = texture_path.read_bytes()
    blob = bytearray()
    views, accessors = [], []

    def add(values, component_type, kind, target=34962):
        values = np.ascontiguousarray(values)
        blob.extend(b"\0" * (-len(blob) % 4))
        view = len(views)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": values.nbytes, "target": target})
        blob.extend(values.tobytes())
        index = len(accessors)
        accessors.append({"bufferView": view, "componentType": component_type,
                          "count": len(values), "type": kind})
        return index

    # Two coincident triangles at different depths. The +Z view must show face 1.
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                          [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype="<f4")
    # A constant (u=0, v=0) samples the bottom-left texel in glTF UV space.
    uv = np.zeros((6, 2), dtype="<f4")
    indices = np.array([[0], [1], [2], [3], [4], [5]], dtype="<u2")
    pos_idx = add(positions, 5126, "VEC3")
    uv_idx = add(uv, 5126, "VEC2")
    index_idx = add(indices, 5123, "SCALAR", 34963)
    blob.extend(b"\0" * (-len(blob) % 4))
    image_view = len(views)
    views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(png_bytes)})
    blob.extend(png_bytes)
    doc = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{
            "attributes": {"POSITION": pos_idx, "TEXCOORD_0": uv_idx},
            "indices": index_idx, "material": 0,
        }]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}], "images": [{"bufferView": image_view, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(blob)}], "bufferViews": views, "accessors": accessors,
    }
    write_glb(path, doc, blob)


class RenderViewsTests(unittest.TestCase):
    def test_texture_depth_visibility_and_source_face_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            _textured_glb(source)
            rgb, instances, faces = render_view(source, "+Z", resolution=80)
            covered = instances >= 0
            self.assertGreater(covered.sum(), 100)
            self.assertEqual(set(np.unique(faces[covered])), {1})
            self.assertEqual(set(np.unique(instances[covered])), {0})
            np.testing.assert_array_equal(rgb[covered], np.broadcast_to([15, 40, 230], rgb[covered].shape))

    def test_six_view_outputs_include_rgb_and_id_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scene.glb"
            output = Path(directory) / "views"
            _textured_glb(source)
            render_views(source, output, resolution=(48, 32))
            metadata = json.loads((output / "views.json").read_text())
            self.assertEqual(metadata["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(metadata["views"]["+X"], {"rgb": "view_px.png", "ids": "view_px.npz"})
            for axis in ("px", "mx", "py", "my", "pz", "mz"):
                self.assertTrue((output / f"view_{axis}.png").is_file())
                with np.load(output / f"view_{axis}.npz") as arrays:
                    self.assertEqual(arrays["source_instance"].shape, (32, 48))
                    self.assertEqual(arrays["source_face"].shape, (32, 48))


if __name__ == "__main__":
    unittest.main()
