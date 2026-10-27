import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infer_labels import infer_source_faces


class LocalInferenceTests(unittest.TestCase):
    def test_fuses_two_views_and_rejects_stale_source_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scene.glb"
            source.write_bytes(b"source one")
            views = root / "views"
            views.mkdir()
            ids = np.full((2, 4), -1, dtype=np.int32)
            ids[:, :2] = 3
            faces = np.full((2, 4), -1, dtype=np.int32)
            faces[:, :2] = 7
            metadata = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "views": {}}
            for stem in ("a", "b"):
                Image.new("RGB", (4, 2), (100, 100, 100)).save(views / f"{stem}.png")
                np.savez_compressed(views / f"{stem}.npz", source_instance=ids, source_face=faces)
                metadata["views"][stem] = {"rgb": f"{stem}.png", "ids": f"{stem}.npz"}
            (views / "views.json").write_text(json.dumps(metadata))
            predicted = [(np.array([[19, 19, 0, 0], [19, 19, 0, 0]], dtype=np.uint8),
                          np.full((2, 4), .9, dtype=np.float32)),
                         (np.array([[19, 19, 0, 0], [19, 0, 0, 0]], dtype=np.uint8),
                          np.full((2, 4), .9, dtype=np.float32))]
            with patch("infer_labels._load_model", return_value=(object(), {}, {19: "chair", 0: "wall"})), \
                 patch("infer_labels._predict_image", side_effect=predicted):
                result = infer_source_faces(source, views, root, root / "labels.json",
                                            min_pixels=4, min_agreement=.7)
            self.assertEqual(len(result["labeled_faces"]), 1)
            label = result["labeled_faces"][0]
            self.assertEqual((label["source_instance"], label["source_face"]), (3, 7))
            self.assertEqual(label["semantic_class"], "chair")
            self.assertEqual(label["object_name"], "chair_source_0003")
            source.write_bytes(b"source two")
            with self.assertRaisesRegex(ValueError, "do not match"):
                infer_source_faces(source, views, root, root / "labels.json")


if __name__ == "__main__":
    unittest.main()
