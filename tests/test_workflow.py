import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from collision import export_collisions, raycast_scene, raycast_triangles
from segment_objects import load_geometry, segment


class RayTests(unittest.TestCase):
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    faces = np.array([[0, 1, 2]])

    def test_nonunit_direction_and_both_sides(self):
        for origin, direction in [([.5, .5, 3], [0, 0, -7]), ([.5, .5, -3], [0, 0, 7])]:
            hit = raycast_triangles(self.vertices, self.faces, origin, direction)
            self.assertAlmostEqual(hit['distance_source_units'], 3)
            np.testing.assert_allclose(hit['position_world'], [.5, .5, 0])

    def test_misses_and_distance_limit(self):
        self.assertIsNone(raycast_triangles(self.vertices, self.faces, [3, 3, 1], [0, 0, -1]))
        self.assertIsNone(raycast_triangles(self.vertices, self.faces, [.5, .5, 1], [1, 0, 0]))
        self.assertIsNone(raycast_triangles(self.vertices, self.faces, [.5, .5, 3], [0, 0, -1], 2))

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            raycast_triangles(self.vertices, self.faces, [0, 0, 1], [0, 0, 0])
        with self.assertRaises(ValueError):
            raycast_triangles(self.vertices, self.faces, [0, 0, 1], [0, np.nan, -1])

    def test_nearest_object_id_is_independent_of_file_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            objects = []
            for object_id, z in [(10, 0), (42, 2)]:
                filename = f'{object_id}.npz'
                np.savez(folder / filename, vertices=self.vertices + [0, 0, z], faces=self.faces,
                         source_mesh=np.array([0]), source_face=np.array([123]))
                objects.append(dict(id=object_id, name=str(object_id), semantic_class='building',
                                    status='test', npz=filename))
            scene = folder / 'scene.json'
            scene.write_text(json.dumps({'objects': objects}))
            hit = raycast_scene(scene, [.5, .5, 5], [0, 0, -3])
            self.assertEqual(hit['object_id'], 42)
            self.assertEqual(hit['source_face'], 123)
            self.assertAlmostEqual(hit['distance_source_units'], 3)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.results = Path(temporary.name) / 'results'
        source = ROOT / 'data/destroyed_apartments_bucha_ukraine_war_2022.glb'
        segment(source, cls.results, ROOT / 'annotations.json')
        export_collisions(source, cls.results)

    def test_default_annotation_counts_and_partition(self):
        with np.load(self.results / 'face-labels.npz') as data:
            self.assertEqual(len(data['object_id']), 183051)
            self.assertEqual(np.bincount(data['object_id']).tolist(), [0, 78967, 21408, 21280, 18976, 37877, 4543])
            keys = np.column_stack([data['source_mesh'], data['source_face']])
            self.assertEqual(len(np.unique(keys, axis=0)), 183051)

    def test_collision_files_retain_exact_world_triangles(self):
        _, _, chunks = load_geometry(ROOT / 'data/destroyed_apartments_bucha_ukraine_war_2022.glb')
        scene = json.loads((self.results / 'collision-scene.json').read_text())
        for obj in scene['objects']:
            with np.load(self.results / obj['npz']) as data:
                actual = data['vertices'][data['faces']]
                self.assertEqual(len(actual), obj['triangles'])
                for chunk in chunks:
                    selected = data['source_mesh'] == chunk['mesh']
                    expected = chunk['tri'][data['source_face'][selected]]
                    np.testing.assert_array_equal(actual[selected], expected)

    def test_actual_scan_rays_identify_distinct_objects(self):
        scene = self.results / 'collision-scene.json'
        rubble = raycast_scene(scene, [10, 0, 20], [0, 0, -1])
        building = raycast_scene(scene, [7, 8, 20], [0, 0, -1])
        self.assertEqual(rubble['object_id'], 5)
        self.assertEqual(building['object_id'], 1)


if __name__ == '__main__':
    unittest.main()
