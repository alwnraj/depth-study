"""Export per-object collision surfaces and query their IDs entirely in Python.

This is a double-sided triangle ray query, not a robot dynamics simulator.
Distances are in the source coordinate system; meters per unit are unknown.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from segment_objects import load_geometry


def export_collisions(input_glb, result_directory):
    output = Path(result_directory)
    manifest = json.loads((output / 'manifest.json').read_text())
    _, _, chunks = load_geometry(input_glb)
    triangles = np.concatenate([c['tri'] for c in chunks])
    with np.load(output / 'face-labels.npz') as data:
        labels = data['object_id']
        source_arrays = {
            name: data[name]
            for name in ('source_instance', 'source_node', 'source_mesh', 'source_primitive', 'source_face')
            if name in data.files
        }
    if len(labels) != len(triangles):
        raise ValueError('Source geometry and label counts do not match.')
    folder = output / 'colliders'
    folder.mkdir(exist_ok=True)
    objects = []
    for obj in manifest['objects']:
        selected = labels == obj['id']
        vertices, inverse = np.unique(triangles[selected].reshape(-1, 3), axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3).astype(np.uint32)
        # Welding identical positions is safe for geometry-only collision meshes.
        # Triangle winding, positions and face-to-object membership remain exact.
        if not np.array_equal(vertices[faces], triangles[selected]):
            raise RuntimeError('Collision export changed the source triangle geometry.')
        npz_name = f"colliders/{obj['name']}.npz"
        obj_name = f"colliders/{obj['name']}.obj"
        np.savez_compressed(output / npz_name, vertices=vertices, faces=faces,
                            **{name: values[selected] for name, values in source_arrays.items()},
                            object_id=np.int32(obj['id']))
        with (output / obj_name).open('w') as stream:
            stream.write('# Collision surface: input GLB world coordinates; orientation and scale are unverified\n')
            stream.write('o ' + obj['name'] + '\n')
            np.savetxt(stream, vertices, fmt='v %.17g %.17g %.17g')
            np.savetxt(stream, faces.astype(np.int64) + 1, fmt='f %d %d %d')
        objects.append({k: obj[k] for k in ['id', 'name', 'semantic_class', 'status', 'triangles']} | {
            'npz': npz_name, 'obj': obj_name, 'collision_type': 'static_triangle_surface',
            'simplified': False, 'closed_solid': False})
    scene = {'coordinate_frame': manifest.get('coordinate_frame', 'Input GLB world coordinates'),
             'orientation': manifest.get('orientation'), 'meters_per_unit': manifest.get('meters_per_source_unit'),
             'objects': objects, 'source_sha256': manifest['source_sha256'],
             'geometry_verified_exact': True,
             'limitations': ['Open scan surfaces retain holes and incomplete backs',
                             'No dynamics, robot footprint, clearance or navigation planner',
                             'Semantic labels are provisional; ground does not mean traversable']}
    (output / 'collision-scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    return scene


def derive_verification_ray(scene_path):
    """Choose a face-derived ray that can verify one generated object ID."""
    scene_path = Path(scene_path)
    scene = json.loads(scene_path.read_text())
    attempts = []
    for obj in scene['objects']:
        with np.load(scene_path.parent / obj['npz']) as data:
            if len(data['faces']) == 0:
                continue
            triangles = data['vertices'][data['faces']]
            cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            areas = np.linalg.norm(cross, axis=1)
            for face_index in np.argsort(areas)[::-1][:8]:
                if areas[face_index] <= 1e-12:
                    continue
                normal = cross[face_index] / areas[face_index]
                center = triangles[face_index].mean(axis=0)
                span = max(float(np.ptp(data['vertices'], axis=0).max()), 1.0)
                offset = max(span * 1e-5, 1e-4)
                attempts.append({
                    'origin': (center + normal * offset).tolist(),
                    'direction': (-normal).tolist(),
                    'expected_object_id': int(obj['id']),
                    'expected_object_name': obj['name'],
                })
    for attempt in attempts:
        hit = raycast_scene(scene_path, attempt['origin'], attempt['direction'])
        if hit and hit['object_id'] == attempt['expected_object_id']:
            attempt['hit'] = hit
            return attempt
    return None


def normalize_ray(origin, direction, max_distance):
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite([origin, direction]).all():
        raise ValueError('Ray origin and direction must be finite three-dimensional vectors.')
    length = np.linalg.norm(direction)
    if length == 0 or not np.isfinite(length):
        raise ValueError('Ray direction must be nonzero and finite.')
    if np.isnan(max_distance) or max_distance <= 0:
        raise ValueError('Maximum ray distance must be positive.')
    return origin, direction / length


def raycast_triangles(vertices, faces, origin, direction, max_distance=np.inf):
    origin, direction = normalize_ray(origin, direction, max_distance)
    if len(faces) == 0:
        return None
    tri = vertices[faces]
    edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum('ij,ij->i', edge1, p)
    usable = np.abs(determinant) > 1e-12
    inv = np.zeros_like(determinant)
    inv[usable] = 1 / determinant[usable]
    offset = origin - tri[:, 0]
    u = np.einsum('ij,ij->i', offset, p) * inv
    q = np.cross(offset, edge1)
    v = (q @ direction) * inv
    distance = np.einsum('ij,ij->i', edge2, q) * inv
    hit = usable & (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
    hit &= (distance > 1e-8) & (distance <= max_distance)
    if not hit.any():
        return None
    index = int(np.argmin(np.where(hit, distance, np.inf)))
    return {'distance_source_units': float(distance[index]), 'collision_face': index,
            'position_world': (origin + direction * distance[index]).tolist()}


def raycast_scene(scene_path, origin, direction, max_distance=np.inf):
    scene_path = Path(scene_path)
    origin, direction = normalize_ray(origin, direction, max_distance)
    scene = json.loads(scene_path.read_text())
    nearest = None
    limit = max_distance
    for obj in scene['objects']:
        with np.load(scene_path.parent / obj['npz']) as data:
            hit = raycast_triangles(data['vertices'], data['faces'], origin, direction, limit)
            if hit is None:
                continue
            index = hit['collision_face']
            hit.update(object_id=obj['id'], object_name=obj['name'], semantic_class=obj['semantic_class'],
                       label_status=obj['status'], source_mesh=int(data['source_mesh'][index]),
                       source_face=int(data['source_face'][index]))
            for name in ('source_instance', 'source_node', 'source_primitive'):
                if name in data.files:
                    hit[name] = int(data[name][index])
            limit = hit['distance_source_units']
            nearest = hit
    return nearest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=Path(__file__).with_name('results') / 'collision-scene.json')
    parser.add_argument('--origin', nargs=3, type=float, required=True, metavar=('X', 'Y', 'Z'))
    parser.add_argument('--direction', nargs=3, type=float, required=True, metavar=('DX', 'DY', 'DZ'))
    parser.add_argument('--max-distance', type=float, default=np.inf)
    args = parser.parse_args()
    try:
        result = raycast_scene(args.scene, args.origin, args.direction, args.max_distance)
    except (ValueError, OSError) as error:
        parser.error(str(error) + ' Run run.py first if the collision scene does not exist.')
    print(json.dumps({'hit': result, 'units': 'source units; real-world scale unverified'}, indent=2))


if __name__ == '__main__':
    main()
