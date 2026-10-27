import io
import unittest
import numpy as np
from PIL import Image
from depth_demo_server import estimate


class FakeSession:
    def run(self, names, inputs):
        self.tensor = inputs['pixel_values']
        return [np.tile(np.linspace(0, 1, 28, dtype=np.float32), (1, 14, 1))]


class FakeMetricSession:
    def get_inputs(self):
        return [type('Input', (), {'name': 'input'})()]

    def run(self, names, inputs):
        self.tensor = inputs['input']
        return [np.tile(np.array([2.0, 10.0], dtype=np.float32), (1, 2, 1))]


class DemoTests(unittest.TestCase):
    def test_image_to_comparison_and_surface_data(self):
        image = Image.new('RGB', (120, 60), (100, 150, 200))
        stream = io.BytesIO()
        image.save(stream, 'PNG')
        session = FakeSession()
        result = estimate(stream.getvalue(), session)
        self.assertEqual(result['original_size'], [120, 60])
        self.assertEqual(session.tensor.shape[0:2], (1, 3))
        self.assertEqual(len(result['near']), result['width'] * result['height'])
        self.assertLess(result['near'][0], result['near'][result['width'] - 1])
        self.assertTrue(result['image'] and result['depth'] and result['colors'])

    def test_metric_values_preserve_meters_and_image_positions(self):
        stream = io.BytesIO()
        Image.new('RGB', (120, 60), (100, 150, 200)).save(stream, 'PNG')
        metric = FakeMetricSession()
        result = estimate(stream.getvalue(), FakeSession(), metric, 'outdoor')
        self.assertEqual(metric.tensor.shape, (1, 3, 518, 518))
        self.assertEqual(result['metric']['scene'], 'outdoor')
        self.assertAlmostEqual(result['metric']['values'][0], 2.0)
        self.assertAlmostEqual(result['metric']['values'][255], 10.0)
        self.assertEqual(len(result['metric']['values']), 256 * 256)

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(OSError):
            estimate(b'not an image', FakeSession())


if __name__ == '__main__':
    unittest.main()
