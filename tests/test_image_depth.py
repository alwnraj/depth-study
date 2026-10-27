import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from image_depth import preprocess, visualization, run


class ImageDepthTests(unittest.TestCase):
    def test_preprocessing_shape_and_rgb_normalization(self):
        image = Image.new('RGB', (640, 320), (255, 0, 0))
        tensor, size = preprocess(image)
        self.assertEqual(size, (518, 252))
        self.assertEqual(tensor.shape, (1, 3, 252, 518))
        np.testing.assert_allclose(tensor[0, :, 0, 0],
                                   [(1 - .485) / .229, - .456 / .224, - .406 / .225], rtol=1e-6)

    def test_visualization_is_near_white_and_constant_safe(self):
        raw = np.array([[1, 3]], dtype=np.float32)
        result = np.asarray(visualization(raw, (2, 1)))
        np.testing.assert_array_equal(result, [[0, 255]])
        self.assertFalse(np.asarray(visualization(np.ones((1, 2), dtype=np.float32), (2, 1))).any())

    def test_output_restricted_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'input.png'
            Image.new('RGB', (2, 2)).save(image)
            with self.assertRaises(ValueError):
                run(image, Path(tmp) / 'results-test')
            self.assertTrue(image.exists())


if __name__ == '__main__':
    unittest.main()
