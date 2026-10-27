import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collision import derive_verification_ray, export_collisions
from object_inventory import connected_surfaces, inventory_labels
from segment_objects import accessor, load_geometry, read_glb, segment
from test_synthetic_glb import _synthetic_glb


class ConnectivityTests(unittest.TestCase):
    def test_split_one_chunk_join_across_chunks_without_merging_point_contacts(self):
        first = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        adjacent = np.array([[1., 0, 0], [1, 1, 0], [0, 1, 0]])
        distant = first + [5, 0, 0]
        point_contact = np.array([[1., 1, 0], [2, 1, 0], [2, 2, 0]])
        near = first + [0, 0, 1e-7]
        # A single source chunk can contain distinct objects; an adjacent surface
        # can live in a different chunk (or duplicate UV vertices at its seam).
        chunks = [{'source_instance': 8, 'tri': np.array([first, distant]), 'faces': np.zeros((2, 3))},
                  {'source_instance': 3, 'tri': np.array([adjacent, point_contact, near]),
                   'faces': np.zeros((3, 3))}]
        labels, confidence, review, objects, _ = inventory_labels(chunks, 'a' * 64)
        self.assertEqual(labels[0], labels[2])
        self.assertEqual(len(np.unique(labels)), 4)
        self.assertNotEqual(labels[0], labels[1])
        self.assertNotEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[4])
        self.assertTrue(np.all(confidence == 0))
        self.assertTrue(review.all())
        self.assertEqual({obj['semantic_class'] for obj in objects}, {'unknown'})
        reversed_objects = inventory_labels(chunks[::-1], 'a' * 64)[3]
        self.assertEqual({obj['stable_id'] for obj in objects},
                         {obj['stable_id'] for obj in reversed_objects})
        changed_source = inventory_labels(chunks, 'b' * 64)[3]
        self.assertTrue({obj['stable_id'] for obj in objects}.isdisjoint(
            {obj['stable_id'] for obj in changed_source}))

    def test_degenerate_faces_do_not_bridge_components_and_nonmanifold_edges_join(self):
        tri = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]],
                        [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
                        [[0, 0, 0], [1, 0, 0], [0, -1, 0]],
                        [[0, 0, 0], [1, 0, 0], [1, 0, 0]]])
        labels = connected_surfaces(tri)
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertNotEqual(labels[0], labels[3])
        np.testing.assert_array_equal(connected_surfaces(tri[-1:]), [1])
        with self.assertRaisesRegex(ValueError, 'finite'):
            connected_surfaces(tri * np.nan)
        with self.assertRaisesRegex(ValueError, 'at least one'):
            connected_surfaces(np.empty((0, 3, 3)))


class InventoryExportTests(unittest.TestCase):
    def test_inventory_bounds_identity_face_ranges_textures_and_collisions(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source, output = folder / 'synthetic.glb', folder / 'results'
            _synthetic_glb(source)
            before = source.read_bytes()
            manifest = segment(source, output, inventory=True)
            inventory = json.loads((output / 'inventory.json').read_text())
            _, _, chunks = load_geometry(source)
            originals = {chunk['source_instance']: chunk for chunk in chunks}
            self.assertEqual(inventory['source_sha256'], hashlib.sha256(before).hexdigest())
            self.assertEqual(inventory['verified_physical_object_count'], 0)
            self.assertEqual(inventory['known_semantic_label_count'], 0)
            self.assertEqual(inventory['source_triangles'], 29)
            # Ground crosses two primitives, so five chunks become four candidates.
            self.assertEqual(inventory['candidate_count'], 4)
            self.assertIsNone(inventory['coordinate_frame']['meters_per_source_unit'])
            self.assertTrue(all(manifest['validation'].values()))
            self.assertFalse(manifest['validated_semantic_accuracy'])
            self.assertEqual(list(output.glob('*.glb')), [output / 'segmented-scene.glb'])
            with np.load(output / 'face-labels.npz') as mapping:
                all_refs = []
                for candidate in inventory['candidates']:
                    refs = []
                    for ref in candidate['source_faces']:
                        for start, end in ref['face_ranges_inclusive']:
                            refs.extend((ref['source_instance'], face) for face in range(start, end + 1))
                        chunk = originals[ref['source_instance']]
                        for key in ('source_node', 'source_mesh', 'source_primitive'):
                            self.assertEqual(ref[key], chunk[key])
                    all_refs.extend(refs)
                    selected = mapping['object_id'] == candidate['object_id']
                    npz_refs = list(zip(mapping['source_instance'][selected], mapping['source_face'][selected]))
                    self.assertEqual(refs, npz_refs)
                    triangles = np.array([originals[instance]['tri'][face] for instance, face in refs])
                    bounds = [triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))]
                    np.testing.assert_array_equal(candidate['bounds_3d']['min'], bounds[0])
                    np.testing.assert_array_equal(candidate['bounds_3d']['max'], bounds[1])
                    np.testing.assert_array_equal(candidate['position_xyz'], np.mean(bounds, axis=0))
                    self.assertFalse(candidate['uncertainty']['physical_object_verified'])
                    self.assertIsNone(candidate['uncertainty']['position_error_source_units'])
                self.assertEqual(len(all_refs), len(set(all_refs)))
                self.assertEqual(len(all_refs), 29)

            # Independently check the textured export's original attribute values
            # and embedded image bytes, in addition to the exporter's own checks.
            source_doc, source_blob = read_glb(source)
            export_doc, export_blob = read_glb(output / 'segmented-scene.glb')
            self.assertEqual(source_doc['materials'], export_doc['materials'])
            for node in export_doc['nodes']:
                if 'mesh' not in node:
                    continue
                original = originals[node['extras']['sourceInstance']]
                primitive = export_doc['meshes'][node['mesh']]['primitives'][0]
                for name, index in primitive['attributes'].items():
                    actual = accessor(export_doc, export_blob, index)
                    expected = accessor(source_doc, source_blob, original['primitive']['attributes'][name])
                    self.assertTrue(all(any(np.array_equal(value, row) for row in expected) for value in actual))
            for doc, blob in ((source_doc, source_blob), (export_doc, export_blob)):
                view = doc['bufferViews'][doc['images'][0]['bufferView']]
                image = blob[view.get('byteOffset', 0):view.get('byteOffset', 0) + view['byteLength']]
                if doc is source_doc:
                    original_image = image
                else:
                    self.assertEqual(image, original_image)

            scene = export_collisions(source, output)
            self.assertIsNone(scene['meters_per_unit'])
            for obj in scene['objects']:
                with np.load(output / obj['npz']) as collision:
                    expected = np.array([originals[int(instance)]['tri'][face]
                                         for instance, face in zip(collision['source_instance'], collision['source_face'])])
                    np.testing.assert_array_equal(collision['vertices'][collision['faces']], expected)
            ray = derive_verification_ray(output / 'collision-scene.json')
            self.assertIsNotNone(ray)
            self.assertEqual(ray['expected_object_id'], ray['hit']['object_id'])
            self.assertEqual(ray['hit']['semantic_class'], 'unknown')
            self.assertEqual(source.read_bytes(), before)

            second = segment(source, folder / 'second', inventory=True)
            self.assertEqual([obj['stable_id'] for obj in manifest['objects']],
                             [obj['stable_id'] for obj in second['objects']])
            # Reusing the directory for an older mode must remove inventory data.
            segment(source, output)
            self.assertFalse((output / 'inventory.json').exists())
            self.assertFalse((output / scene['objects'][0]['npz']).exists())

    def test_inventory_rejects_other_label_modes(self):
        with self.assertRaisesRegex(ValueError, 'separately'):
            segment('unused.glb', 'unused-output', annotations='unused.json', inventory=True)


if __name__ == '__main__':
    unittest.main()
