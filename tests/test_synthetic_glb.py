import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collision import export_collisions
from segment_objects import UnsupportedGLBError, load_geometry, read_glb, segment, write_glb


PNG_1X1 = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')


def _synthetic_glb(path):
    blob = bytearray()
    buffer_views = []
    accessors = []

    def add_array(values, component_type, value_type, target):
        values = np.ascontiguousarray(values)
        blob.extend(b'\0' * (-len(blob) % 4))
        view_index = len(buffer_views)
        buffer_views.append({'buffer': 0, 'byteOffset': len(blob),
                             'byteLength': values.nbytes, 'target': target})
        blob.extend(values.tobytes())
        accessor_index = len(accessors)
        accessors.append({'bufferView': view_index, 'componentType': component_type,
                          'count': len(values), 'type': value_type})
        return accessor_index

    def primitive(positions, indices):
        positions = np.asarray(positions, dtype='<f4')
        normals = np.zeros_like(positions)
        normals[:, 2] = 1
        uv = np.zeros((len(positions), 2), dtype='<f4')
        uv[:, 0] = np.linspace(0, 1, len(positions))
        return {
            'attributes': {
                'POSITION': add_array(positions, 5126, 'VEC3', 34962),
                'NORMAL': add_array(normals, 5126, 'VEC3', 34962),
                'TEXCOORD_0': add_array(uv, 5126, 'VEC2', 34962),
            },
            'indices': add_array(np.asarray(indices, dtype='<u2').reshape(-1, 1), 5123, 'SCALAR', 34963),
            'material': 0,
        }

    def cube(width, depth, height):
        return [
            [0, 0, 0], [width, 0, 0], [width, depth, 0], [0, depth, 0],
            [0, 0, height], [width, 0, height], [width, depth, height], [0, depth, height],
        ], [
            0, 2, 1, 0, 3, 2,
            4, 5, 6, 4, 6, 7,
            0, 1, 5, 0, 5, 4,
            1, 2, 6, 1, 6, 5,
            2, 3, 7, 2, 7, 6,
            3, 0, 4, 3, 4, 7,
        ]

    ground_a = primitive([[0, 0, 0], [6, 0, 0], [6, 8, 0], [0, 8, 0]],
                         [0, 1, 2, 0, 2, 3])
    ground_b = primitive([[6, 0, 0], [12, 0, 0], [12, 8, 0], [6, 8, 0]],
                         [0, 1, 2, 0, 2, 3])
    building_a_positions, building_a_indices = cube(2, 2, 4)
    building_a = primitive(building_a_positions, building_a_indices)
    building_b_positions, building_b_indices = cube(2, 2, 3)
    building_b = primitive(building_b_positions, building_b_indices)
    rubble = primitive([[0, 0, 0.10], [1, 0, 0.35], [0.5, 1, 0.20]], [0, 1, 2])

    blob.extend(b'\0' * (-len(blob) % 4))
    image_view = len(buffer_views)
    buffer_views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(PNG_1X1)})
    blob.extend(PNG_1X1)

    document = {
        'asset': {'version': '2.0', 'generator': 'synthetic loader regression'},
        'scene': 0,
        'scenes': [{'name': 'synthetic outdoor test', 'nodes': [0, 5]}],
        'nodes': [
            {'name': 'translated_tilted_world', 'translation': [2, -1, 0],
             'rotation': [0.1564345, 0, 0, 0.9876883], 'children': [1, 2, 4]},
            {'name': 'ground_mesh', 'mesh': 0},
            {'name': 'building_parent', 'rotation': [0, 0, 0.0871557, 0.9961947], 'children': [3]},
            {'name': 'building_a', 'mesh': 1, 'translation': [1, 1, 0]},
            {'name': 'rubble', 'mesh': 3, 'translation': [7, 3, 0]},
            {'name': 'second_world', 'children': [6]},
            {'name': 'building_b', 'mesh': 2, 'translation': [9, 5, 0]},
        ],
        'meshes': [
            {'name': 'ground_with_two_primitives', 'primitives': [ground_a, ground_b]},
            {'name': 'building_a_mesh', 'primitives': [building_a]},
            {'name': 'building_b_mesh', 'primitives': [building_b]},
            {'name': 'rubble_mesh', 'primitives': [rubble]},
        ],
        'materials': [{'name': 'test_material', 'pbrMetallicRoughness': {
            'baseColorTexture': {'index': 0}, 'metallicFactor': 0, 'roughnessFactor': 1}}],
        'textures': [{'sampler': 0, 'source': 0}],
        'samplers': [{'magFilter': 9729, 'minFilter': 9729}],
        'images': [{'bufferView': image_view, 'mimeType': 'image/png'}],
        'buffers': [{'byteLength': len(blob)}],
        'bufferViews': buffer_views,
        'accessors': accessors,
    }
    write_glb(path, document, blob)


class SyntheticGLBTests(unittest.TestCase):
    def test_loader_and_automatic_pipeline_handle_scene_graph_variation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / 'synthetic.glb'
            output = folder / 'results'
            _synthetic_glb(source)

            document, blob, chunks = load_geometry(source)
            self.assertEqual(len(chunks), 5)
            self.assertEqual(len({chunk['source_root'] for chunk in chunks}), 2)
            self.assertEqual(len({chunk['source_mesh'] for chunk in chunks}), 4)
            self.assertEqual({chunk['source_primitive'] for chunk in chunks if chunk['source_mesh'] == 0}, {0, 1})
            np.testing.assert_allclose(chunks[0]['tri'][0, 0], [2, -1, 0])
            self.assertEqual(document['images'][0]['mimeType'], 'image/png')
            image_view = document['bufferViews'][document['images'][0]['bufferView']]
            start = image_view.get('byteOffset', 0)
            self.assertEqual(blob[start:start + image_view['byteLength']], PNG_1X1)

            manifest = segment(source, output)
            export_collisions(source, output)
            self.assertEqual(manifest['method'], 'automatic_geometry_proposal')
            self.assertEqual(manifest['source_geometry']['triangles'], 29)
            self.assertFalse(manifest['validated_semantic_accuracy'])
            classes = {obj['semantic_class'] for obj in manifest['objects']}
            self.assertTrue({'building', 'ground_road', 'rubble', 'unknown'} <= classes)
            self.assertTrue(manifest['validation']['all_source_faces_partitioned_once'])
            self.assertTrue(manifest['validation']['original_materials_and_embedded_images_exact'])
            with np.load(output / 'face-labels.npz') as labels:
                self.assertEqual(len(labels['object_id']), 29)
                self.assertEqual(len(np.unique(np.column_stack([
                    labels['source_instance'], labels['source_face']] ), axis=0)), 29)
            scene = json.loads((output / 'collision-scene.json').read_text())
            for obj in scene['objects']:
                with np.load(output / obj['npz']) as collision:
                    self.assertIn('source_instance', collision.files)

            document, binary = read_glb(source)
            document['meshes'][0]['primitives'][0]['mode'] = 1
            invalid = folder / 'invalid-mode.glb'
            write_glb(invalid, document, binary)
            with self.assertRaisesRegex(UnsupportedGLBError, 'only TRIANGLES'):
                load_geometry(invalid)

            unsupported_features = {
                'animations': lambda doc: doc.update({'animations': [{'channels': [], 'samplers': []}]}),
                'skins': lambda doc: doc.update({'skins': [{'joints': [1]}]}),
                'morph targets': lambda doc: doc['meshes'][0]['primitives'][0].update({'targets': [{}]}),
            }
            for feature, change in unsupported_features.items():
                feature_document, feature_binary = read_glb(source)
                change(feature_document)
                unsupported = folder / f'unsupported-{feature.replace(" ", "-")}.glb'
                write_glb(unsupported, feature_document, feature_binary)
                with self.assertRaisesRegex(UnsupportedGLBError, feature):
                    load_geometry(unsupported)

            with np.load(output / 'face-labels.npz') as labels:
                source_face = np.flatnonzero((labels['source_instance'] == 0)
                                             & (labels['source_face'] == 0))[0]
                predicted_id = int(labels['object_id'][source_face])
            predicted_object = next(obj for obj in manifest['objects'] if obj['id'] == predicted_id)
            review_file = folder / 'review.json'
            review_file.write_text(json.dumps({
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'reviewed_faces': [{'source_instance': 0, 'source_face': 0,
                                    'expected_instance_id': predicted_object['name'],
                                    'semantic_class': predicted_object['semantic_class']}],
            }))
            reviewed = segment(source, folder / 'reviewed-results', None, review_file)
            self.assertEqual(reviewed['method'], 'automatic_geometry_proposal')
            self.assertTrue(reviewed['evaluation']['available'])
            self.assertTrue(reviewed['evaluation']['valid'])
            self.assertEqual(reviewed['evaluation']['samples'], 1)
            self.assertEqual(reviewed['evaluation']['object_id_accuracy'], 1.0)
            self.assertEqual(reviewed['evaluation']['instance_accuracy'], 1.0)
            self.assertEqual(reviewed['evaluation']['semantic_class_accuracy'], 1.0)
            self.assertTrue(reviewed['evaluation']['valid_automatic_review'])
            self.assertTrue(reviewed['validated_semantic_accuracy'])

            invalid_review_file = folder / 'invalid-review.json'
            invalid_review_file.write_text(json.dumps({
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'reviewed_faces': [
                    {'source_instance': 0, 'source_face': 0,
                     'expected_instance_id': predicted_object['name'],
                     'semantic_class': predicted_object['semantic_class']},
                    {'source_instance': 999, 'source_face': 0,
                     'semantic_class': predicted_object['semantic_class']},
                ],
            }))
            invalid_review = segment(source, folder / 'invalid-review-results', None, invalid_review_file)
            self.assertFalse(invalid_review['validated_semantic_accuracy'])
            self.assertFalse(invalid_review['evaluation']['valid'])
            self.assertEqual(invalid_review['evaluation']['requested_samples'], 2)
            self.assertEqual(invalid_review['evaluation']['samples'], 1)
            self.assertEqual(invalid_review['evaluation']['unmatched_samples'], 1)
            self.assertIsNone(invalid_review['evaluation']['semantic_class_accuracy'])

    def test_face_corrections_preserve_source_mapping_and_evaluation_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / 'synthetic.glb'
            _synthetic_glb(source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            baseline = segment(source, folder / 'baseline')
            with np.load(folder / 'baseline' / 'face-labels.npz') as labels:
                source_arrays = {
                    name: labels[name].copy()
                    for name in ('source_instance', 'source_node', 'source_mesh',
                                 'source_primitive', 'source_face')
                }
                original_ids = labels['object_id'].copy()

            corrections = folder / 'corrections.json'
            corrections.write_text(json.dumps({
                'source_sha256': source_hash,
                'corrected_instances': [{
                    'source_instance': 0,
                    'object_name': 'reviewed_component',
                    'semantic_class': 'building',
                }],
                'corrected_faces': [{
                    'source_instance': 0,
                    'source_face': 0,
                    'object_name': 'manual_ground_patch',
                    'semantic_class': 'ground_road',
                }],
            }))
            corrected = segment(source, folder / 'corrected', corrections=corrections)
            manual_object = next(obj for obj in corrected['objects']
                                 if obj['name'] == 'manual_ground_patch')
            component_object = next(obj for obj in corrected['objects']
                                    if obj['name'] == 'reviewed_component')
            self.assertEqual(manual_object['semantic_class'], 'ground_road')
            self.assertEqual(manual_object['triangles'], 1)
            self.assertEqual(component_object['semantic_class'], 'building')
            self.assertEqual(component_object['triangles'], int(np.sum(source_arrays['source_instance'] == 0)) - 1)
            self.assertEqual(corrected['labeling']['manual_correction_count'],
                             int(np.sum(source_arrays['source_instance'] == 0)))
            self.assertFalse(corrected['validated_semantic_accuracy'])
            self.assertFalse(corrected['evaluation']['available'])
            for invariant in ('original_vertex_attributes_exact',
                              'original_materials_and_embedded_images_exact',
                              'world_space_triangle_coordinates_exact',
                              'source_transforms_preserved_for_output_roots'):
                self.assertTrue(corrected['validation'][invariant])

            with np.load(folder / 'corrected' / 'face-labels.npz') as labels:
                self.assertEqual(int(labels['object_id'][0]), manual_object['id'])
                self.assertEqual(len(labels['object_id']), len(original_ids))
                for name, expected in source_arrays.items():
                    np.testing.assert_array_equal(labels[name], expected)
                changed = labels['object_id'] != original_ids
                np.testing.assert_array_equal(
                    np.flatnonzero(changed),
                    np.flatnonzero(source_arrays['source_instance'] == 0))

            export_collisions(source, folder / 'corrected')
            scene = json.loads((folder / 'corrected' / 'collision-scene.json').read_text())
            collision_object = next(obj for obj in scene['objects']
                                    if obj['name'] == 'manual_ground_patch')
            with np.load(folder / 'corrected' / collision_object['npz']) as collision:
                self.assertEqual(int(collision['object_id']), manual_object['id'])
                np.testing.assert_array_equal(collision['source_instance'], [0])
                np.testing.assert_array_equal(collision['source_face'], [0])

            mismatched = folder / 'wrong-source-corrections.json'
            mismatched.write_text(json.dumps({
                'source_sha256': '0' * 64,
                'corrected_faces': [],
            }))
            with self.assertRaisesRegex(ValueError, 'source_sha256 does not match'):
                segment(source, folder / 'wrong-source-results', corrections=mismatched)

    def test_imported_labels_start_unknown_and_preserve_source_faces(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / 'synthetic.glb'
            _synthetic_glb(source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            baseline = segment(source, folder / 'baseline')
            with np.load(folder / 'baseline' / 'face-labels.npz') as prior:
                refs = {name: prior[name].copy() for name in
                        ('source_instance', 'source_node', 'source_mesh',
                         'source_primitive', 'source_face')}

            label_file = folder / 'labels.json'
            label_file.write_text(json.dumps({
                'source_sha256': source_hash,
                'labeled_faces': [{
                    'source_instance': 0, 'source_face': 0,
                    'object_name': 'chair', 'semantic_class': 'furniture',
                    'confidence': 0.72,
                }],
            }))
            imported = segment(source, folder / 'imported', labels=label_file)
            self.assertEqual(imported['method'], 'imported_source_face_labels')
            self.assertFalse(imported['validated_semantic_accuracy'])
            self.assertFalse(imported['evaluation']['available'])
            chair = next(obj for obj in imported['objects'] if obj['name'] == 'chair')
            unknown = next(obj for obj in imported['objects'] if obj['name'] == 'unknown')
            self.assertEqual(chair['semantic_class'], 'furniture')
            self.assertEqual(chair['triangles'], 1)
            self.assertEqual(unknown['triangles'], imported['source_geometry']['triangles'] - 1)
            with np.load(folder / 'imported' / 'face-labels.npz') as output:
                self.assertEqual(int(output['object_id'][0]), chair['id'])
                self.assertAlmostEqual(float(output['label_confidence'][0]), 0.72)
                self.assertTrue(np.all(output['object_id'][1:] == unknown['id']))
                self.assertTrue(np.all(output['boundary_review'][1:] == 1))
                for name, expected in refs.items():
                    np.testing.assert_array_equal(output[name], expected)

            export_collisions(source, folder / 'imported')
            scene = json.loads((folder / 'imported' / 'collision-scene.json').read_text())
            chair_collider = next(obj for obj in scene['objects'] if obj['name'] == 'chair')
            with np.load(folder / 'imported' / chair_collider['npz']) as collision:
                np.testing.assert_array_equal(collision['source_instance'], [0])
                np.testing.assert_array_equal(collision['source_face'], [0])

            wrong_hash = folder / 'wrong-labels.json'
            wrong_hash.write_text(json.dumps({'source_sha256': '0' * 64, 'labeled_faces': []}))
            with self.assertRaisesRegex(ValueError, 'source_sha256 does not match'):
                segment(source, folder / 'wrong-label-results', labels=wrong_hash)

    def test_manual_corrections_drop_objects_with_no_faces(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / 'synthetic.glb'
            _synthetic_glb(source)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            _, _, chunks = load_geometry(source)
            correction_file = folder / 'all-building.json'
            correction_file.write_text(json.dumps({
                'source_sha256': source_hash,
                'corrected_instances': [{
                    'source_instance': chunk['source_instance'],
                    'object_name': 'reviewed_building',
                    'semantic_class': 'building',
                } for chunk in chunks],
                'corrected_faces': [],
            }))
            manifest = segment(source, folder / 'all-building-results', corrections=correction_file)
            self.assertEqual([obj['name'] for obj in manifest['objects']], ['reviewed_building'])
            self.assertEqual(manifest['objects'][0]['triangles'], 29)


if __name__ == '__main__':
    unittest.main()
