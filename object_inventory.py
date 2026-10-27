"""Conservative, offline 3D surface candidates; no semantic recognition model."""
import hashlib
import json

import numpy as np


METHOD = 'shared_world_edges_v1'
LIMITATIONS = [
    'Connected surfaces are candidates, not verified physical objects.',
    'Touching or reconstructed bridges can join several objects, walls and floors.',
    'Holes, disconnected parts and unequal seam tessellation can split one object.',
    'Only exactly equal world-space edge endpoints are joined; no gap closing is used.',
    'Source names are provenance only; all semantic labels remain unknown.',
    'XYZ is the observed axis-aligned bounds center, not a physical object origin.',
    'Bounds describe observed surfaces, not complete solids; position error is unknown.',
    'Coordinates retain source units and axes; physical scale and robot registration are unknown.',
]


def connected_surfaces(triangles):
    """Join faces sharing an entire exact edge across all chunks, including UV seams.

    Point contact and proximity alone do not join faces. Zero-area triangles are
    retained as separate candidates, so they cannot bridge otherwise separate parts.
    Geometry is never welded or modified; welding here is only an adjacency test.
    """
    triangles = np.asarray(triangles)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError('Inventory requires at least one triangle with XYZ vertices.')
    if not np.isfinite(triangles).all():
        raise ValueError('Inventory requires finite world-space positions.')
    count = len(triangles)
    _, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    vertices = inverse.reshape(-1, 3)
    edges = np.sort(np.concatenate([vertices[:, [0, 1]], vertices[:, [1, 2]],
                                    vertices[:, [2, 0]]]), axis=1)
    faces = np.tile(np.arange(count), 3)
    nondegenerate = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1) > 0
    usable = np.tile(nondegenerate, 3)
    edges, faces = edges[usable], faces[usable]
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, faces = edges[order], faces[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    parent = np.arange(count)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in zip(faces[:-1][same], faces[1:][same]):
        first, second = find(first), find(second)
        if first != second:
            parent[max(first, second)] = min(first, second)
    roots = np.array([find(index) for index in range(count)])
    _, labels = np.unique(roots, return_inverse=True)
    return (labels + 1).astype(np.uint32)


def inventory_labels(chunks, source_hash):
    """Return candidates through the existing labeling/export interface."""
    triangles = np.concatenate([chunk['tri'] for chunk in chunks])
    labels = connected_surfaces(triangles)
    source_keys = np.concatenate([
        np.column_stack((np.full(len(chunk['faces']), chunk['source_instance']),
                         np.arange(len(chunk['faces'])))) for chunk in chunks
    ]).astype('<u8')
    objects = []
    for object_id in np.unique(labels):
        # Identity depends on this exact source and the candidate's face membership,
        # not its bounds, semantic name, output directory or traversal of groups.
        digest = hashlib.sha256(bytes.fromhex(source_hash))
        keys = source_keys[labels == object_id]
        keys = keys[np.lexsort((keys[:, 1], keys[:, 0]))]
        digest.update(keys.tobytes())
        stable_id = 'candidate_' + digest.hexdigest()[:24]
        objects.append({'id': int(object_id), 'name': stable_id,
                        'stable_id': stable_id, 'semantic_class': 'unknown',
                        'status': 'Unverified connected-surface candidate'})
    if len({obj['stable_id'] for obj in objects}) != len(objects):
        raise AssertionError('Candidate ID collision.')
    return (labels, np.zeros(len(labels)), np.ones(len(labels), dtype=bool), objects,
            {'mode': 'offline_3d_inventory', 'grouping_method': METHOD,
             'edge_tolerance_source_units': 0, 'limitations': LIMITATIONS})


def write_inventory(output, manifest, chunks, labels):
    """Write self-describing candidates with compact, inclusive source-face ranges.

    face-labels.npz is the full per-face mapping shared with collision exports.
    JSON ranges provide the same references without repeating every face as JSON.
    """
    source_hash = manifest['source_sha256']
    frame = {'id': 'glb_world_' + source_hash,
             'axes': ['X', 'Y', 'Z'], 'units': 'source GLB units',
             'transforms': 'All active scene node transforms applied; no recentering or rotation',
             'meters_per_source_unit': None, 'robot_registration': None}
    candidates = []
    for obj in manifest['objects']:
        refs = []
        cursor = 0
        for chunk in chunks:
            count = len(chunk['faces'])
            selected = np.flatnonzero(labels[cursor:cursor + count] == obj['id'])
            cursor += count
            if not len(selected):
                continue
            breaks = np.flatnonzero(np.diff(selected) > 1)
            starts = np.r_[selected[0], selected[breaks + 1]]
            ends = np.r_[selected[breaks], selected[-1]]
            refs.append({name: int(chunk[name]) for name in
                         ('source_instance', 'source_node', 'source_mesh', 'source_primitive')} | {
                             'face_ranges_inclusive': np.column_stack((starts, ends)).tolist()})
        bounds = np.asarray(obj['bounds_world'])
        fraction = obj['triangles'] / manifest['source_geometry']['triangles']
        reasons = ['Connectivity alone does not establish a physical object boundary.',
                   'May contain several touching objects or only a disconnected part of one object.']
        if fraction > 0.5:
            reasons.append('Dominant scene component; likely unresolved scene structure or multiple objects.')
        if obj['triangles'] <= 10:
            reasons.append('Small surface fragment (10 faces or fewer); may be reconstruction debris.')
        candidates.append({
            'id': obj['stable_id'], 'object_id': obj['id'],
            'semantic_label': obj['semantic_class'], 'status': obj['status'],
            'position_xyz': ((bounds[0] + bounds[1]) / 2).tolist(),
            'position_definition': 'axis_aligned_observed_bounds_center',
            'bounds_3d': {'min': bounds[0].tolist(), 'max': bounds[1].tolist()},
            'coordinate_frame': frame['id'], 'triangles': obj['triangles'],
            'scene_face_fraction': fraction, 'source_mesh_occurrences': len(refs),
            'uncertainty': {'physical_object_verified': False, 'semantic_label_verified': False,
                            'grouping': 'unverified', 'bounds_complete': False,
                            'position_error_source_units': None, 'reasons': reasons},
            'source_faces': refs,
            'face_mapping': {'file': 'face-labels.npz', 'select_object_id': obj['id']},
            'visual_asset': {'file': obj['asset'], 'object_id': obj['id']},
        })
    inventory = {
        'schema_version': 1, 'source_file': manifest['source_file'],
        'source_sha256': source_hash, 'grouping_method': METHOD,
        'id_stability': 'Same source bytes and source-face membership; not persistent across edited GLBs',
        'coordinate_frame': frame, 'candidate_count': len(candidates),
        'verified_physical_object_count': 0, 'known_semantic_label_count': 0,
        'source_triangles': len(labels), 'candidates': candidates, 'limitations': LIMITATIONS,
    }
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return inventory
