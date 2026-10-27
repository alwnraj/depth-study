import unittest

import numpy as np

from live_depth import depth_display, model_input


class LiveDepthTests(unittest.TestCase):
    def test_frame_preprocessing_matches_each_model(self):
        frame = np.zeros((40, 80, 3), dtype=np.uint8)
        frame[:, :, 2] = 255  # OpenCV BGR red
        relative = model_input(frame, 'relative')
        metric = model_input(frame, 'indoor')
        self.assertEqual(relative.shape, (1, 3, 252, 518))
        self.assertEqual(metric.shape, (1, 3, 518, 518))
        self.assertGreater(float(metric[0, 0, 0, 0]), 2.0)
        self.assertLess(float(metric[0, 2, 0, 0]), -1.0)

    def test_near_is_white_for_relative_and_metric_outputs(self):
        relative = np.array([[[0., 1.]]], dtype=np.float32)
        metric = np.array([[[1., 10.]]], dtype=np.float32)
        # np.squeeze removes dimensions from 1-pixel height: use 2 rows instead.
        relative = np.repeat(relative, 2, axis=1)
        metric = np.repeat(metric, 2, axis=1)
        np.testing.assert_array_equal(depth_display(relative, 'relative', (2, 2))[0], [0, 255])
        np.testing.assert_array_equal(depth_display(metric, 'outdoor', (2, 2))[0], [255, 0])

    def test_reject_nonfinite_output(self):
        with self.assertRaises(ValueError):
            depth_display(np.full((1, 2, 2), np.nan), 'relative', (2, 2))


if __name__ == '__main__':
    unittest.main()
