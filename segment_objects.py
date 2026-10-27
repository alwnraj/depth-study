#!/usr/bin/env python3
"""Load GLB scenes, propose face labels and export textured object assets.

The loader accepts ordinary embedded glTF 2.0 triangle scenes with multiple
nodes, meshes, primitives and transforms. It keeps local vertex attributes
unchanged and evaluates every face in the input world coordinate frame.

Annotations remain supported for the Bucha regression case. When no
annotation file is supplied, a geometry-only proposal labels ground, low
debris, elevated building-like surfaces and unknown geometry. These labels
are provisional and are not a trained semantic model.
"""
import argparse
import copy
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np


class GLBError(ValueError):
    """The input is not a supported, readable GLB scene."""


class UnsupportedGLBError(GLBError):
    """The GLB is valid but uses a feature this Python loader does not handle."""


COMPONENT_DTYPES = {
    5120: np.dtype('<i1'),
    5121: np.dtype('u1'),
    5122: np.dtype('<i2'),
    5123: np.dtype('<u2'),
    5125: np.dtype('<u4'),
    5126: np.dtype('<f4'),
}
TYPE_WIDTHS = {
    'SCALAR': 1,
    'VEC2': 2,
    'VEC3': 3,
    'VEC4': 4,
    'MAT2': 4,
    'MAT3': 9,
    'MAT4': 16,
}
TRIANGLES_MODE = 4
PALETTE = ['#5489df', '#b577d8', '#e5a245', '#58b99d', '#d77b62', '#9ba5ad',
           '#c46b9c', '#78a85b', '#d2a84c', '#7e8ed4']


def _json_error(path, detail):
    return GLBError(f'{Path(path).name}: {detail}')


def read_glb(path):
    """Read a GLB and return its JSON document plus embedded binary bytes."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 12:
        raise _json_error(path, 'file is shorter than the GLB header.')
    magic, version, length = struct.unpack_from('<III', raw, 0)
    if magic != 0x46546C67 or version != 2:
        raise _json_error(path, 'expected a glTF 2.0 binary (GLB) file.')
    if length != len(raw):
        raise _json_error(path, f'header length {length} does not match file length {len(raw)}.')

    offset = 12
    document = None
    binary = None
    while offset < length:
        if offset + 8 > length:
            raise _json_error(path, 'truncated chunk header.')
        chunk_length, chunk_type = struct.unpack_from('<II', raw, offset)
        start = offset + 8
        end = start + chunk_length
        if end > length:
            raise _json_error(path, 'chunk extends past the end of the file.')
        payload = raw[start:end]
        offset = end
        if chunk_type == 0x4E4F534A:
            if document is not None:
                raise _json_error(path, 'contains more than one JSON chunk.')
            try:
                document = json.loads(payload)
            except json.JSONDecodeError as error:
                raise _json_error(path, f'JSON chunk is invalid: {error.msg}.') from error
        elif chunk_type == 0x004E4942:
            if binary is not None:
                raise _json_error(path, 'contains more than one binary chunk.')
            binary = payload

    if document is None:
        raise _json_error(path, 'does not contain a JSON chunk.')
    binary = binary or b''
    _validate_supported_document(path, document, binary)
    return document, binary


def _validate_supported_document(path, document, binary):
    if document.get('skins'):
        raise UnsupportedGLBError(
            f'{Path(path).name}: skins are not supported because object export rewrites scene node indices.')
    if document.get('animations'):
        raise UnsupportedGLBError(
            f'{Path(path).name}: animations are not supported because object export rewrites scene node and accessor indices.')
    for node_index, node in enumerate(document.get('nodes', [])):
        if node.get('skin') is not None:
            raise UnsupportedGLBError(
                f'{Path(path).name}: node {node_index} uses a skin; skinned nodes are not supported by object export.')
        if node.get('weights'):
            raise UnsupportedGLBError(
                f'{Path(path).name}: node {node_index} uses morph weights; morph targets are not supported by object export.')
    buffers = document.get('buffers', [])
    if len(buffers) > 1:
        raise UnsupportedGLBError(f'{Path(path).name}: multiple buffers are not supported.')
    if buffers:
        buffer = buffers[0]
        if buffer.get('uri'):
            raise UnsupportedGLBError(f'{Path(path).name}: external buffer URIs are not supported; embed the buffer in the GLB.')
        if buffer.get('byteLength', 0) > len(binary):
            raise _json_error(path, 'buffer byteLength exceeds the binary chunk.')
    for index, buffer_view in enumerate(document.get('bufferViews', [])):
        if buffer_view.get('buffer', 0) != 0:
            raise UnsupportedGLBError(f'{Path(path).name}: bufferView {index} does not use buffer 0.')
    for index, accessor_def in enumerate(document.get('accessors', [])):
        if accessor_def.get('sparse'):
            raise UnsupportedGLBError(f'{Path(path).name}: sparse accessor {index} is not supported.')
    for index, image in enumerate(document.get('images', [])):
        uri = image.get('uri')
        if uri and not uri.startswith('data:'):
            raise UnsupportedGLBError(f'{Path(path).name}: external image URI {index} is not supported; embed image bytes in the GLB.')
    for mesh_index, mesh in enumerate(document.get('meshes', [])):
        if mesh.get('weights'):
            raise UnsupportedGLBError(
                f'{Path(path).name}: mesh {mesh_index} uses morph weights; morph targets are not supported by object export.')
        for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
            mode = primitive.get('mode', TRIANGLES_MODE)
            if mode != TRIANGLES_MODE:
                raise UnsupportedGLBError(
                    f'{Path(path).name}: mesh {mesh_index} primitive {primitive_index} uses mode {mode}; only TRIANGLES (4) is supported.')
            if 'extensions' in primitive and any(name in primitive['extensions'] for name in
                                                  ('KHR_draco_mesh_compression', 'EXT_meshopt_compression')):
                raise UnsupportedGLBError(
                    f'{Path(path).name}: mesh {mesh_index} primitive {primitive_index} uses compressed geometry.')
            if primitive.get('targets'):
                raise UnsupportedGLBError(
                    f'{Path(path).name}: mesh {mesh_index} primitive {primitive_index} uses morph targets; object export cannot preserve their references.')


def accessor(doc, blob, index, decode_normalized=False):
    """Return an accessor as a copied ``(count, width)`` NumPy array."""
    if index not in range(len(doc.get('accessors', []))):
        raise GLBError(f'Accessor index {index} is out of range.')
    definition = doc['accessors'][index]
    if definition.get('sparse'):
        raise UnsupportedGLBError(f'Accessor {index} uses sparse storage, which is not supported.')
    component_type = definition.get('componentType')
    if component_type not in COMPONENT_DTYPES:
        raise UnsupportedGLBError(f'Accessor {index} uses unsupported component type {component_type}.')
    value_type = definition.get('type')
    if value_type not in TYPE_WIDTHS:
        raise UnsupportedGLBError(f'Accessor {index} uses unsupported type {value_type}.')
    if 'bufferView' not in definition:
        raise UnsupportedGLBError(f'Accessor {index} has no bufferView; sparse or zero-filled accessors are not supported.')
    view_index = definition['bufferView']
    if view_index not in range(len(doc.get('bufferViews', []))):
        raise GLBError(f'Accessor {index} references missing bufferView {view_index}.')
    view = doc['bufferViews'][view_index]
    dtype = COMPONENT_DTYPES[component_type]
    width = TYPE_WIDTHS[value_type]
    element_bytes = dtype.itemsize * width
    stride = view.get('byteStride', element_bytes)
    if stride < element_bytes or stride % dtype.itemsize:
        raise GLBError(f'Accessor {index} has an invalid byteStride.')
    offset = view.get('byteOffset', 0) + definition.get('byteOffset', 0)
    count = definition.get('count', 0)
    required = offset + (max(count - 1, 0) * stride) + element_bytes
    if required > len(blob):
        raise GLBError(f'Accessor {index} extends past the embedded binary data.')
    values = np.ndarray((count, width), dtype=dtype, buffer=blob, offset=offset,
                        strides=(stride, dtype.itemsize)).copy()
    if decode_normalized and definition.get('normalized'):
        values = _decode_normalized(values, component_type)
    return values


def _decode_normalized(values, component_type):
    if component_type == 5120:
        return np.maximum(values.astype(np.float32) / 127.0, -1.0)
    if component_type == 5121:
        return values.astype(np.float32) / 255.0
    if component_type == 5122:
        return np.maximum(values.astype(np.float32) / 32767.0, -1.0)
    if component_type == 5123:
        return values.astype(np.float32) / 65535.0
    if component_type == 5125:
        return values.astype(np.float64) / 4294967295.0
    return values.astype(np.float32)


def _node_matrix(node):
    if 'matrix' in node:
        matrix = np.asarray(node['matrix'], dtype=float)
        if matrix.shape != (16,):
            raise GLBError('A node matrix must contain 16 values.')
        return matrix.reshape(4, 4, order='F')
    translation = np.asarray(node.get('translation', [0, 0, 0]), dtype=float)
    scale = np.asarray(node.get('scale', [1, 1, 1]), dtype=float)
    rotation = np.asarray(node.get('rotation', [0, 0, 0, 1]), dtype=float)
    if translation.shape != (3,) or scale.shape != (3,) or rotation.shape != (4,):
        raise GLBError('Node translation, rotation and scale must have valid glTF dimensions.')
    if not np.isfinite(np.concatenate([translation, scale, rotation])).all():
        raise GLBError('Node transforms must be finite.')
    length = np.linalg.norm(rotation)
    if length == 0:
        raise GLBError('A node quaternion cannot be zero.')
    x, y, z, w = rotation / length
    rotation_matrix = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
        [0, 0, 0, 1],
    ], dtype=float)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix[:3, :3] @ np.diag(scale)
    transform[:3, 3] = translation
    return transform


def _matrix_to_gltf(matrix):
    return np.asarray(matrix, dtype=float).reshape(4, 4).flatten(order='F').tolist()


def _transform_positions(positions, matrix):
    homogeneous = np.ones((len(positions), 4), dtype=float)
    homogeneous[:, :3] = positions
    transformed = homogeneous @ matrix.T
    if np.any(np.abs(transformed[:, 3] - 1) > 1e-8):
        transformed[:, :3] /= transformed[:, 3:4]
    return transformed[:, :3]


def _scene_roots(doc):
    nodes = doc.get('nodes', [])
    if not nodes:
        raise GLBError('The GLB contains no nodes; a scene graph is required.')
    scenes = doc.get('scenes', [])
    if scenes:
        scene_index = doc.get('scene', 0)
        if scene_index not in range(len(scenes)):
            raise GLBError(f'Active scene index {scene_index} is out of range.')
        roots = scenes[scene_index].get('nodes', [])
    else:
        children = {child for node in nodes for child in node.get('children', [])}
        roots = [index for index in range(len(nodes)) if index not in children]
    if not roots:
        raise GLBError('The active GLB scene has no root nodes.')
    return roots


def load_geometry(path):
    """Load every triangle primitive reachable from the active GLB scene.

    Each returned chunk represents one node/mesh/primitive occurrence. Its
    local attributes remain available through ``primitive`` and its world
    triangles are in the input scene coordinate frame. ``source_instance``,
    ``source_node``, ``source_mesh``, ``source_primitive`` and ``source_face``
    together identify the original face.
    """
    doc, blob = read_glb(path)
    nodes = doc['nodes']
    chunks = []
    roots = _scene_roots(doc)
    occurrence = 0

    def walk(node_index, parent_world, root_index, root_world, path_indices):
        nonlocal occurrence
        if node_index not in range(len(nodes)):
            raise GLBError(f'Node index {node_index} is out of range.')
        if node_index in path_indices:
            raise GLBError(f'Node cycle detected at node {node_index}.')
        node = nodes[node_index]
        world = parent_world @ _node_matrix(node)
        if 'mesh' in node:
            mesh_index = node['mesh']
            if mesh_index not in range(len(doc.get('meshes', []))):
                raise GLBError(f'Node {node_index} references missing mesh {mesh_index}.')
            mesh = doc['meshes'][mesh_index]
            for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
                attributes = primitive.get('attributes', {})
                if 'POSITION' not in attributes:
                    raise UnsupportedGLBError(
                        f'Node {node_index} mesh {mesh_index} primitive {primitive_index} has no POSITION attribute.')
                positions = accessor(doc, blob, attributes['POSITION'])
                if positions.shape[1] != 3 or doc['accessors'][attributes['POSITION']]['componentType'] != 5126:
                    raise UnsupportedGLBError(
                        f'Node {node_index} mesh {mesh_index} primitive {primitive_index} POSITION must be float VEC3.')
                if 'indices' in primitive:
                    index_values = accessor(doc, blob, primitive['indices']).ravel()
                    if len(index_values) % 3:
                        raise GLBError(
                            f'Node {node_index} mesh {mesh_index} primitive {primitive_index} index count is not divisible by 3.')
                    faces = index_values.reshape(-1, 3).astype(np.int64, copy=False)
                else:
                    if len(positions) % 3:
                        raise GLBError(
                            f'Node {node_index} mesh {mesh_index} primitive {primitive_index} non-indexed vertex count is not divisible by 3.')
                    faces = np.arange(len(positions), dtype=np.int64).reshape(-1, 3)
                if len(faces) and (faces.min() < 0 or faces.max() >= len(positions)):
                    raise GLBError(
                        f'Node {node_index} mesh {mesh_index} primitive {primitive_index} references a vertex outside POSITION.')
                world_positions = _transform_positions(positions.astype(float), world)
                tri = world_positions[faces]
                cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                area = np.linalg.norm(cross, axis=1) / 2
                normals = cross / np.maximum(area[:, None] * 2, 1e-12)
                relative = np.linalg.inv(root_world) @ world
                chunks.append({
                    'chunk': len(chunks),
                    'source_instance': occurrence,
                    'source_root': root_index,
                    'source_node': node_index,
                    'source_mesh': mesh_index,
                    'mesh': mesh_index,
                    'source_primitive': primitive_index,
                    'primitive': primitive,
                    'faces': faces,
                    'positions': positions,
                    'world_transform': world,
                    'relative_transform': relative,
                    'tri': tri,
                    'centers': tri.mean(axis=1),
                    'area': area,
                    'normals': normals,
                })
                occurrence += 1
        for child in node.get('children', []):
            walk(child, world, root_index, root_world, path_indices | {node_index})

    identity = np.eye(4, dtype=float)
    for root_index in roots:
        root_world = _node_matrix(nodes[root_index])
        walk(root_index, identity, root_index, root_world, set())
    if not chunks:
        raise GLBError('The active GLB scene contains no triangle primitives.')
    return doc, blob, chunks


def write_glb(path, doc, blob):
    blob = bytes(blob) + b'\0' * (-len(blob) % 4)
    doc.setdefault('buffers', [{'byteLength': 0}])
    doc['buffers'][0]['byteLength'] = len(blob)
    payload = json.dumps(doc, separators=(',', ':')).encode()
    payload += b' ' * (-len(payload) % 4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(
        struct.pack('<III', 0x46546C67, 2, 28 + len(payload) + len(blob))
        + struct.pack('<II', len(payload), 0x4E4F534A) + payload
        + struct.pack('<II', len(blob), 0x004E4942) + blob)


def polygon_mask_and_distance(points, polygon, mapping=None):
    """Even-odd containment plus nearest boundary distance in world XY."""
    if mapping is None:
        raise ValueError('Annotation polygons require an explicit polygon_mapping.')
    origin = np.asarray(mapping.get('origin', [0, 0]), dtype=float)
    scale = np.asarray(mapping.get('pixels_per_world_unit', [1, 1]), dtype=float)
    if origin.shape != (2,) or scale.shape != (2,) or np.any(scale == 0):
        raise ValueError('polygon_mapping must provide two nonzero pixels_per_world_unit values and a two-value origin.')
    vertices = origin + np.asarray(polygon, dtype=float) / scale
    inside = np.zeros(len(points), dtype=bool)
    distance = np.full(len(points), np.inf)
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = b - a
        if abs(edge[1]) > 1e-12:
            crossing = ((a[1] > points[:, 1]) != (b[1] > points[:, 1]))
            inside ^= crossing & (points[:, 0] < a[0] + edge[0] * (points[:, 1] - a[1]) / edge[1])
        t = np.clip(((points - a) @ edge) / max(edge @ edge, 1e-12), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(points - a - t[:, None] * edge, axis=1))
    return inside, distance


def _all_face_arrays(chunks):
    return (np.concatenate([chunk['tri'] for chunk in chunks]),
            np.concatenate([chunk['centers'] for chunk in chunks]),
            np.concatenate([chunk['normals'] for chunk in chunks]),
            np.concatenate([chunk['area'] for chunk in chunks]))


def _median_edge_length(triangles):
    edges = np.concatenate([
        np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
        np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
        np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
    ])
    finite = edges[np.isfinite(edges) & (edges > 1e-12)]
    return float(np.median(finite)) if len(finite) else 1.0


def _basis_from_up(up):
    up = np.asarray(up, dtype=float)
    up /= max(np.linalg.norm(up), 1e-12)
    reference = np.array([0., 0., 1.]) if abs(up[2]) < .9 else np.array([1., 0., 0.])
    right = np.cross(reference, up)
    right /= max(np.linalg.norm(right), 1e-12)
    forward = np.cross(up, right)
    forward /= max(np.linalg.norm(forward), 1e-12)
    return right, forward, up


def estimate_ground_plane(chunks):
    """Estimate an up direction and a low, approximately planar surface."""
    triangles, centers, normals, area = _all_face_arrays(chunks)
    valid = np.isfinite(centers).all(axis=1) & (area > 1e-12)
    if valid.sum() < 3:
        raise ValueError('At least three nondegenerate faces are required for automatic orientation estimation.')
    points = centers[valid]
    edge_scale = _median_edge_length(triangles[valid])
    sample = points if len(points) <= 100000 else points[::max(1, len(points) // 100000)]
    covariance = np.cov(sample - sample.mean(axis=0), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    pca_candidate = eigenvectors[:, int(np.argmin(eigenvalues))]
    pca_candidate /= max(np.linalg.norm(pca_candidate), 1e-12)
    normal_indices = np.flatnonzero(valid)
    if len(normal_indices) > 128:
        normal_indices = normal_indices[np.linspace(0, len(normal_indices) - 1, 128, dtype=int)]
    candidates = [pca_candidate]
    for normal in normals[normal_indices]:
        if np.linalg.norm(normal) <= 1e-12:
            continue
        normal = normal / np.linalg.norm(normal)
        if not any(abs(float(normal @ existing)) > .995 for existing in candidates):
            candidates.append(normal)
    best = None
    for candidate in candidates:
        projection = centers @ candidate
        span = float(np.percentile(projection[valid], 98) - np.percentile(projection[valid], 2))
        band = max(span * .015, edge_scale * .5, span * 1e-4, 1e-6)
        for sign in (1., -1.):
            up = candidate * sign
            projected = centers @ up
            low = float(np.percentile(projected[valid], 8))
            near = valid & (np.abs(projected - low) <= band)
            aligned = near & (np.abs(normals @ up) >= np.cos(np.deg2rad(35)))
            alignment = np.abs(normals[aligned] @ up)
            score = float((area[aligned] * alignment).sum())
            if best is None or score > best[0]:
                best = score, up, projected, near, aligned, band
    _, up, projected, near, aligned, band = best
    plane_value = float(np.median(projected[aligned])) if aligned.any() else float(np.percentile(projected[valid], 8))
    offset = -plane_value
    residual_vertices = triangles @ up + offset
    residual_centers = residual_vertices.mean(axis=1)
    ground_band = max(band, edge_scale * .5)
    ground = (valid & (np.abs(residual_centers) <= ground_band)
              & (np.max(np.abs(residual_vertices), axis=1) <= ground_band * 1.75)
              & (np.abs(normals @ up) >= np.cos(np.deg2rad(35))))
    confidence = min(1.0, float(area[ground].sum()) / max(float(area[valid].sum()) * .15, 1e-12))
    right, forward, up = _basis_from_up(up)
    return {
        'normal': up,
        'offset': offset,
        'ground_band': ground_band,
        'scene_height': max(float(np.percentile(residual_centers[valid], 98)), ground_band),
        'edge_scale': edge_scale,
        'confidence': confidence,
        'ground_mask': ground,
        'residual_vertices': residual_vertices,
        'residual_centers': residual_centers,
        'right': right,
        'forward': forward,
        'up': up,
    }


def _union_find_components(points, mask, right, forward, grid_size):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    horizontal = np.column_stack([points @ right, points @ forward])
    keys = np.floor(horizontal[indices] / grid_size).astype(np.int64)
    cells = {}
    for face_index, key in zip(indices, keys):
        cells.setdefault((int(key[0]), int(key[1])), []).append(int(face_index))
    cell_keys = list(cells)
    parent = {key: key for key in cell_keys}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(left, right_key):
        left, right_key = find(left), find(right_key)
        if left != right_key:
            parent[right_key] = left

    for key in cell_keys:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor = (key[0] + dx, key[1] + dy)
                if neighbor in parent:
                    union(key, neighbor)
    components = {}
    for key, face_indices in cells.items():
        components.setdefault(find(key), []).extend(face_indices)
    return list(components.values())


def automatic_labels(chunks):
    """Propose geometry-only labels and dynamic object instances."""
    triangles, centers, normals, area = _all_face_arrays(chunks)
    estimate = estimate_ground_plane(chunks)
    residual_centers = estimate['residual_centers']
    up = estimate['up']
    ground_band = estimate['ground_band']
    scene_height = estimate['scene_height']
    edge_scale = estimate['edge_scale']
    ground = estimate['ground_mask'] & (area > 1e-12)
    building_minimum = max(ground_band * .75, scene_height * .10)
    high = residual_centers > building_minimum
    building_like = high & ((np.abs(normals @ up) < .80) | (residual_centers > scene_height * .35))
    rubble_maximum = max(ground_band * 2.5, scene_height * .24)
    rubble = (~ground & ~building_like & (residual_centers >= -ground_band * 2)
              & (residual_centers <= rubble_maximum) & (area > 1e-12))

    labels = np.zeros(len(centers), dtype=np.uint32)
    confidence = np.full(len(centers), .15, dtype=float)
    review = np.ones(len(centers), dtype=bool)
    building_components = _union_find_components(
        centers, building_like, estimate['right'], estimate['forward'],
        max(edge_scale * 12, scene_height * .02, 1e-6))
    building_components.sort(key=lambda group: (float(centers[group, 0].mean()), float(centers[group, 1].mean())))

    objects = []
    next_id = 1
    for index, group in enumerate(building_components, start=1):
        object_id = next_id
        next_id += 1
        mask = np.zeros(len(labels), dtype=bool)
        mask[group] = True
        labels[mask] = object_id
        height_confidence = np.clip((residual_centers[mask] - building_minimum)
                                    / max(scene_height * .5, ground_band), 0, 1)
        confidence[mask] = np.maximum(.25, height_confidence)
        objects.append({
            'id': object_id,
            'name': f'building_{index:02d}_auto',
            'semantic_class': 'building',
            'color': PALETTE[(object_id - 1) % len(PALETTE)],
            'status': 'Automatic geometry proposal; building instance is provisional',
        })

    ground_id = next_id
    next_id += 1
    labels[ground] = ground_id
    ground_confidence = (1 - np.abs(residual_centers[ground]) / max(ground_band, 1e-12))
    ground_confidence *= np.clip((np.abs(normals[ground] @ up) - .5) / .5, 0, 1)
    confidence[ground] = np.maximum(.25, ground_confidence)
    objects.append({
        'id': ground_id, 'name': 'ground_auto', 'semantic_class': 'ground_road',
        'color': PALETTE[(ground_id - 1) % len(PALETTE)],
        'status': 'Automatic planar-ground proposal; traversability is not established',
    })

    rubble_id = next_id
    next_id += 1
    labels[rubble] = rubble_id
    rubble_confidence = 1 - np.clip(residual_centers[rubble] / max(rubble_maximum, 1e-12), 0, 1)
    confidence[rubble] = np.maximum(.25, rubble_confidence)
    objects.append({
        'id': rubble_id, 'name': 'rubble_auto', 'semantic_class': 'rubble',
        'color': PALETTE[(rubble_id - 1) % len(PALETTE)],
        'status': 'Automatic low-surface proposal; may include vegetation or ground',
    })

    unknown_id = next_id
    labels[labels == 0] = unknown_id
    confidence[labels == unknown_id] = .10
    objects.append({
        'id': unknown_id, 'name': 'unknown_auto', 'semantic_class': 'unknown',
        'color': PALETTE[(unknown_id - 1) % len(PALETTE)],
        'status': 'Automatic fallback for uncertain geometry',
    })
    review &= confidence < .60
    metadata = {
        'mode': 'automatic_geometry_proposal',
        'ground_plane': {
            'normal': estimate['normal'].tolist(),
            'offset': float(estimate['offset']),
            'band': float(ground_band),
            'confidence': float(estimate['confidence']),
        },
        'basis': {
            'right': estimate['right'].tolist(),
            'forward': estimate['forward'].tolist(),
            'up': estimate['up'].tolist(),
        },
        'thresholds': {
            'building_minimum': float(building_minimum),
            'rubble_maximum': float(rubble_maximum),
            'edge_scale': float(edge_scale),
        },
        'building_instance_count': len(building_components),
    }
    return labels, confidence, review, objects, metadata


def _source_hash_check(cfg, source_hash, kind):
    expected = cfg.get('source_sha256')
    if not expected:
        raise ValueError(f'{kind} files must include source_sha256 so they cannot be applied to a different scan.')
    if expected != source_hash:
        raise ValueError(f'{kind} source_sha256 does not match the input GLB.')


def _annotation_source_check(cfg, source_hash):
    _source_hash_check(cfg, source_hash, 'Annotation')


def _review_source_check(cfg, source_hash):
    _source_hash_check(cfg, source_hash, 'Review')


def _correction_source_check(cfg, source_hash):
    _source_hash_check(cfg, source_hash, 'Correction')


def _label_source_check(cfg, source_hash):
    _source_hash_check(cfg, source_hash, 'Label')


def _import_face_labels(chunks, cfg):
    """Import source-face assignments over an all-unknown baseline."""
    assignments = cfg.get('labeled_faces', [])
    if not isinstance(assignments, list):
        raise ValueError('Label file labeled_faces must be a list.')
    unknown = {'id': 1, 'name': 'unknown', 'semantic_class': 'unknown',
               'color': PALETTE[0], 'status': 'Unlabeled source faces'}
    labels = np.ones(sum(len(chunk['faces']) for chunk in chunks), dtype=np.int32)
    confidence = np.zeros(len(labels), dtype=np.float32)
    review = np.ones(len(labels), dtype=bool)
    objects = [unknown]
    _apply_manual_corrections(chunks, labels, objects,
                              {'corrected_faces': assignments})
    assigned = labels != unknown['id']
    confidence[assigned] = 1.0
    face_offsets = {}
    cursor = 0
    for chunk in chunks:
        for face_index in range(len(chunk['faces'])):
            face_offsets[(chunk['source_instance'], face_index)] = cursor + face_index
        cursor += len(chunk['faces'])
    for item in assignments:
        value = item.get('confidence', 1.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError('Label confidence must be a number from 0 to 1.')
        confidence[face_offsets[(item['source_instance'], item['source_face'])]] = value
    for obj in objects[1:]:
        obj['status'] = 'Imported source-face label; semantic accuracy unvalidated'
    metadata = {'mode': 'imported_source_face_labels',
                'imported_label_count': int(assigned.sum()),
                'unlabeled_face_count': int((~assigned).sum())}
    return labels, confidence, review, objects, metadata


def _apply_manual_corrections(chunks, labels, objects, cfg):
    instance_corrections = cfg.get('corrected_instances', [])
    face_corrections = cfg.get('corrected_faces', [])
    if not isinstance(instance_corrections, list):
        raise ValueError('Correction file corrected_instances must be a list.')
    if not isinstance(face_corrections, list):
        raise ValueError('Correction file corrected_faces must be a list.')

    face_lookup = {}
    instance_lookup = {}
    cursor = 0
    for chunk in chunks:
        instance_faces = []
        for face_index in range(len(chunk['faces'])):
            flat_index = cursor + face_index
            key = (chunk['source_instance'], face_index)
            face_lookup.setdefault(key, []).append(flat_index)
            instance_faces.append(flat_index)
        cursor += len(chunk['faces'])
        instance_lookup.setdefault(chunk['source_instance'], []).extend(instance_faces)

    object_by_name = {obj['name']: obj for obj in objects}
    if len(object_by_name) != len(objects):
        raise ValueError('Automatic object names must be unique before corrections can be applied.')
    next_id = max((int(obj['id']) for obj in objects), default=0) + 1

    def correction_object(item):
        nonlocal next_id
        name = item.get('object_name')
        semantic_class = item.get('semantic_class')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', name):
            raise ValueError('Correction object_name must start with a letter and contain only letters, numbers, _ or -.')
        if not isinstance(semantic_class, str) or not semantic_class.strip():
            raise ValueError('Each correction needs a nonempty semantic_class.')
        if name in object_by_name:
            target = object_by_name[name]
            if target.get('semantic_class') != semantic_class:
                raise ValueError(f'Correction object {name} already has semantic class {target.get("semantic_class")!r}.')
        else:
            target = {
                'id': next_id,
                'name': name,
                'semantic_class': semantic_class,
                'color': PALETTE[(next_id - 1) % len(PALETTE)],
                'status': 'Manually corrected from source-face review',
            }
            next_id += 1
            object_by_name[name] = target
            objects.append(target)
        return int(target['id'])

    corrected_face_indices = set()
    seen_instances = set()
    for item in instance_corrections:
        if not isinstance(item, dict):
            raise ValueError('Each corrected_instances entry must be an object.')
        instance = item.get('source_instance')
        if type(instance) is not int or instance < 0:
            raise ValueError('Each instance correction needs a nonnegative integer source_instance value.')
        if instance in seen_instances:
            raise ValueError(f'Source instance {instance} appears more than once in corrected_instances.')
        seen_instances.add(instance)
        face_indices = instance_lookup.get(instance, [])
        if not face_indices:
            raise ValueError(f'Correction source instance {instance} resolves to no faces.')
        object_id = correction_object(item)
        labels[face_indices] = object_id
        corrected_face_indices.update(face_indices)

    seen_faces = set()
    for item in face_corrections:
        if not isinstance(item, dict):
            raise ValueError('Each corrected_faces entry must be an object.')
        instance = item.get('source_instance')
        face = item.get('source_face')
        if type(instance) is not int or type(face) is not int or instance < 0 or face < 0:
            raise ValueError('Each correction needs nonnegative integer source_instance and source_face values.')
        key = (instance, face)
        candidates = face_lookup.get(key, [])
        if len(candidates) != 1:
            raise ValueError(f'Correction source face {key} resolves to {len(candidates)} faces; expected exactly one.')
        if key in seen_faces:
            raise ValueError(f'Correction source face {key} appears more than once.')
        seen_faces.add(key)
        flat_index = candidates[0]
        labels[flat_index] = correction_object(item)
        corrected_face_indices.add(flat_index)
    return len(corrected_face_indices)


def _annotation_labels(chunks, cfg):
    triangles, centers, normals, area = _all_face_arrays(chunks)
    objects = copy.deepcopy(cfg.get('objects', []))
    if not objects:
        raise ValueError('Annotation file must contain at least one object definition.')
    ids = [int(obj['id']) for obj in objects]
    if len(set(ids)) != len(ids) or any(object_id <= 0 for object_id in ids):
        raise ValueError('Annotation object IDs must be unique positive integers.')
    object_by_id = {int(obj['id']): obj for obj in objects}
    unknown = next((object_id for object_id, obj in object_by_id.items()
                    if obj.get('semantic_class') == 'unknown'), max(ids) + 1)
    if unknown not in object_by_id:
        objects.append({'id': unknown, 'name': 'unknown', 'semantic_class': 'unknown',
                        'color': PALETTE[(unknown - 1) % len(PALETTE)],
                        'status': 'Unassigned geometry'})
        object_by_id[unknown] = objects[-1]
    labels = np.full(len(centers), unknown, dtype=np.uint32)
    confidence = np.full(len(centers), .20, dtype=float)
    review = np.zeros(len(centers), dtype=bool)
    regions = {}
    mapping = cfg.get('polygon_mapping')
    if mapping is None and any(obj.get('polygons_pixels') for obj in objects):
        raise ValueError('Annotation polygons require polygon_mapping.')
    for object_id, obj in object_by_id.items():
        mask = np.zeros(len(centers), dtype=bool)
        distance = np.full(len(centers), np.inf)
        for polygon in obj.get('polygons_pixels', []):
            inside, boundary_distance = polygon_mask_and_distance(centers[:, :2], polygon, mapping)
            mask |= inside
            distance = np.minimum(distance, boundary_distance)
        regions[object_id] = mask
        review |= distance < cfg.get('boundary_review_distance', np.inf)
    plane = cfg.get('plane')
    if plane:
        plane_normal = np.asarray(plane['normal'], dtype=float)
        plane_normal /= max(np.linalg.norm(plane_normal), 1e-12)
        plane_offset = float(plane['offset'])
        residual = triangles @ plane_normal + plane_offset
        z_component = plane_normal[2]
        if abs(z_component) < 1e-8:
            raise ValueError('Annotation plane normal must have a nonzero Z component for this regression format.')
        height = residual.mean(axis=1) / z_component
        rubble_ids = [object_id for object_id, obj in object_by_id.items()
                      if obj.get('semantic_class') == 'rubble']
        ground_ids = [object_id for object_id, obj in object_by_id.items()
                      if obj.get('semantic_class') in ('ground', 'ground_road')]
        building_ids = [object_id for object_id, obj in object_by_id.items()
                        if obj.get('semantic_class') == 'building']
        rubble_region = np.zeros(len(labels), dtype=bool)
        for object_id in rubble_ids:
            rubble_region |= regions[object_id]
        if rubble_ids:
            rubble = rubble_region & (height > cfg.get('rubble_min_height', -.15))
            rubble &= height < cfg.get('rubble_max_height', np.inf)
            labels[rubble] = rubble_ids[0]
            confidence[rubble] = .90
        if ground_ids:
            ground = ((np.abs(residual.mean(axis=1)) <= cfg.get('ground_centroid_distance', np.inf))
                      & (np.max(np.abs(residual), axis=1) <= cfg.get('ground_vertex_distance', np.inf))
                      & (np.abs(normals @ plane_normal) >= np.cos(np.deg2rad(
                          cfg.get('ground_max_normal_angle_degrees', 90))))
                      & (area > 1e-12))
            labels[ground & ~rubble_region] = ground_ids[0]
            confidence[labels == ground_ids[0]] = .90
        if building_ids:
            building_surface = ((height > cfg.get('building_height_minimum', -np.inf))
                                | ((height > cfg.get('building_low_face_minimum', -np.inf))
                                   & (np.abs(normals[:, 2]) < cfg.get('building_low_face_max_abs_normal_z', 1))))
            for object_id in reversed(building_ids):
                selected = regions[object_id] & building_surface
                labels[selected] = object_id
                confidence[selected] = .90
            review |= np.abs(height - cfg.get('building_height_minimum', -np.inf)) < cfg.get('height_review_distance', .08)
    else:
        for object_id, obj in object_by_id.items():
            if obj.get('semantic_class') != 'unknown':
                labels[regions[object_id]] = object_id
                confidence[regions[object_id]] = .90
    confidence[review] = np.minimum(confidence[review], .50)
    metadata = {
        'mode': 'annotation_guided_regression',
        'annotation_source_sha256': cfg['source_sha256'],
        'ground_plane': cfg.get('plane'),
        'basis': None,
    }
    return labels, confidence, review, objects, metadata


def _evaluate_reviewed_faces(review_cfg, review_path, chunks, labels, objects, labeling_method):
    reviewed = review_cfg.get('reviewed_faces', []) if review_cfg else []
    if not isinstance(reviewed, list):
        raise ValueError('Review file reviewed_faces must be a list.')
    result = {
        'available': bool(reviewed),
        'valid': False,
        'requested_samples': len(reviewed),
        'samples': 0,
        'unmatched_samples': len(reviewed),
        'object_id_accuracy': None,
        'instance_accuracy': None,
        'semantic_class_accuracy': None,
        'labeling_method': labeling_method,
        'valid_automatic_review': False,
        'scope': 'reviewed faces for this input only',
        'source': str(Path(review_path).resolve()) if review_path else None,
    }
    if not reviewed:
        return result
    object_classes = {int(obj['id']): obj.get('semantic_class') for obj in objects}
    object_names = {int(obj['id']): obj.get('name') for obj in objects}
    lookup = {}
    cursor = 0
    for chunk in chunks:
        for face_index in range(len(chunk['faces'])):
            flat_index = cursor + face_index
            lookup.setdefault((chunk['source_instance'], face_index), []).append(flat_index)
            lookup.setdefault((chunk['source_node'], chunk['source_mesh'], chunk['source_primitive'], face_index), []).append(flat_index)
            lookup.setdefault((chunk['source_mesh'], chunk['source_primitive'], face_index), []).append(flat_index)
        cursor += len(chunk['faces'])
    object_correct = 0
    object_expected = 0
    class_correct = 0
    class_expected = 0
    counted = 0
    for item in reviewed:
        if 'source_face' not in item:
            continue
        if 'source_instance' in item:
            key = (int(item['source_instance']), int(item['source_face']))
        elif 'source_node' in item and 'source_mesh' in item:
            key = (int(item['source_node']), int(item['source_mesh']),
                   int(item.get('source_primitive', 0)), int(item['source_face']))
        elif 'source_mesh' in item:
            key = (int(item['source_mesh']), int(item.get('source_primitive', 0)), int(item['source_face']))
        else:
            continue
        candidates = lookup.get(key, [])
        if len(candidates) != 1:
            continue
        flat_index = candidates[0]
        actual_id = int(labels[flat_index])
        expected_class = item.get('semantic_class', item.get('expected_semantic_class'))
        expected_instance = item.get('expected_instance_id', item.get('instance_id'))
        if expected_instance is None and 'object_id' in item:
            expected_instance = item['object_id']
        if expected_class is None and 'object_id' in item:
            expected_class = object_classes.get(int(item['object_id']))
        if expected_instance is not None:
            object_expected += 1
            if isinstance(expected_instance, str):
                object_correct += object_names.get(actual_id) == expected_instance
            else:
                object_correct += actual_id == int(expected_instance)
        if expected_class is not None:
            class_expected += 1
            class_correct += object_classes.get(actual_id) == expected_class
        counted += 1
    result['samples'] = counted
    result['unmatched_samples'] = len(reviewed) - counted
    complete = counted == len(reviewed) and counted > 0
    if complete and object_expected == len(reviewed):
        result['object_id_accuracy'] = object_correct / object_expected
        result['instance_accuracy'] = result['object_id_accuracy']
    if complete and class_expected == len(reviewed):
        result['semantic_class_accuracy'] = class_correct / class_expected
    result['valid'] = complete and result['semantic_class_accuracy'] is not None
    result['valid_automatic_review'] = (
        result['valid'] and labeling_method == 'automatic_geometry_proposal')
    return result


def _copy_embedded_images(source_doc, source_blob, output_doc, binary):
    output_doc['images'] = copy.deepcopy(source_doc.get('images', []))
    for image in output_doc['images']:
        if 'bufferView' not in image:
            continue
        source_view = source_doc['bufferViews'][image['bufferView']]
        start = source_view.get('byteOffset', 0)
        data = source_blob[start:start + source_view['byteLength']]
        binary.extend(b'\0' * (-len(binary) % 4))
        image['bufferView'] = len(output_doc['bufferViews'])
        output_doc['bufferViews'].append({
            'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data),
        })
        binary.extend(data)


def _add_output_accessor(output_doc, binary, values, original=None, target=34962):
    values = np.ascontiguousarray(values)
    binary.extend(b'\0' * (-len(binary) % 4))
    view_index = len(output_doc['bufferViews'])
    output_doc['bufferViews'].append({
        'buffer': 0, 'byteOffset': len(binary), 'byteLength': values.nbytes, 'target': target,
    })
    binary.extend(values.tobytes())
    if original:
        definition = {
            'bufferView': view_index,
            'componentType': original['componentType'],
            'count': len(values),
            'type': original['type'],
        }
        if original.get('normalized'):
            definition['normalized'] = True
    else:
        definition = {'bufferView': view_index, 'componentType': 5125,
                      'count': values.size, 'type': 'SCALAR'}
    accessor_index = len(output_doc['accessors'])
    output_doc['accessors'].append(definition)
    return accessor_index


def _export_labeled_glb(path, source_doc, source_blob, chunks, objects, metadata):
    """Write an output GLB while retaining source attributes and texture bytes."""
    output_doc = copy.deepcopy(source_doc)
    output_doc['buffers'] = [{'byteLength': 0}]
    output_doc['bufferViews'] = []
    output_doc['accessors'] = []
    output_doc['meshes'] = []
    source_roots = []
    for chunk in chunks:
        if chunk['source_root'] not in source_roots:
            source_roots.append(chunk['source_root'])
    output_doc['nodes'] = []
    root_output_indices = {}
    for source_root in source_roots:
        root = copy.deepcopy(source_doc['nodes'][source_root])
        root.pop('mesh', None)
        root['children'] = []
        root_output_indices[source_root] = len(output_doc['nodes'])
        output_doc['nodes'].append(root)
    source_scene = source_doc.get('scenes', [{}])[source_doc.get('scene', 0)]
    output_doc['scene'] = 0
    output_doc['scenes'] = [{'name': source_scene.get('name', 'Scene'),
                             'nodes': list(root_output_indices.values())}]
    binary = bytearray()
    _copy_embedded_images(source_doc, source_blob, output_doc, binary)
    checks = []
    for object_spec in objects:
        object_id = int(object_spec['id'])
        selected_chunks = [chunk for chunk in chunks
                           if np.any(chunk['labels'] == object_id)]
        for root_index in source_roots:
            root_chunks = [chunk for chunk in selected_chunks if chunk['source_root'] == root_index]
            if not root_chunks:
                continue
            parent_index = len(output_doc['nodes'])
            parent = {
                'name': object_spec['name'], 'children': [],
                'extras': {'objectId': object_id,
                           'semanticClass': object_spec['semantic_class'],
                           'labelStatus': object_spec['status']},
            }
            output_doc['nodes'].append(parent)
            output_doc['nodes'][root_output_indices[root_index]]['children'].append(parent_index)
            for chunk in root_chunks:
                chosen = np.flatnonzero(chunk['labels'] == object_id)
                used, inverse = np.unique(chunk['faces'][chosen], return_inverse=True)
                primitive = copy.deepcopy(chunk['primitive'])
                primitive['attributes'] = {}
                for name, accessor_index in chunk['primitive']['attributes'].items():
                    values = accessor(source_doc, source_blob, accessor_index)[used]
                    primitive['attributes'][name] = _add_output_accessor(
                        output_doc, binary, values, source_doc['accessors'][accessor_index])
                primitive['indices'] = _add_output_accessor(
                    output_doc, binary, inverse.astype('<u4').reshape(-1, 1), target=34963)
                mesh_index = len(output_doc['meshes'])
                output_doc['meshes'].append({
                    'name': f"{object_spec['name']}_chunk_{chunk['source_instance']}",
                    'primitives': [primitive],
                })
                node_index = len(output_doc['nodes'])
                output_doc['nodes'].append({
                    'name': f"{object_spec['name']}_chunk_{chunk['source_instance']}",
                    'mesh': mesh_index,
                    'matrix': _matrix_to_gltf(chunk['relative_transform']),
                    'extras': {
                        'objectId': object_id,
                        'sourceInstance': chunk['source_instance'],
                        'sourceNode': chunk['source_node'],
                        'sourceMesh': chunk['source_mesh'],
                        'sourcePrimitive': chunk['source_primitive'],
                    },
                })
                parent['children'].append(node_index)
                checks.append((mesh_index, node_index, root_output_indices[root_index], chunk, chosen, used, inverse))
    output_doc.setdefault('asset', {})['generator'] = 'Python geometry labeling and exact textured export'
    output_doc.setdefault('extras', {})['semanticSegmentation'] = metadata
    write_glb(path, output_doc, binary)
    check_doc, check_blob = read_glb(path)
    if check_doc.get('materials', []) != source_doc.get('materials', []):
        raise AssertionError('Export changed the material definitions.')
    for source_image, output_image in zip(source_doc.get('images', []), check_doc.get('images', [])):
        if 'bufferView' not in source_image:
            continue
        source_view = source_doc['bufferViews'][source_image['bufferView']]
        output_view = check_doc['bufferViews'][output_image['bufferView']]
        source_bytes = source_blob[source_view.get('byteOffset', 0):source_view.get('byteOffset', 0) + source_view['byteLength']]
        output_bytes = check_blob[output_view.get('byteOffset', 0):output_view.get('byteOffset', 0) + output_view['byteLength']]
        if source_bytes != output_bytes:
            raise AssertionError('Export changed embedded image bytes.')
    for mesh_index, node_index, root_index, chunk, chosen, used, inverse in checks:
        output_primitive = check_doc['meshes'][mesh_index]['primitives'][0]
        actual_indices = accessor(check_doc, check_blob, output_primitive['indices']).ravel()
        if not np.array_equal(actual_indices, inverse.ravel()):
            raise AssertionError('Export changed a primitive index mapping.')
        for name, source_accessor_index in chunk['primitive']['attributes'].items():
            actual = accessor(check_doc, check_blob, output_primitive['attributes'][name])
            expected = accessor(source_doc, source_blob, source_accessor_index)[used]
            if not np.array_equal(actual, expected):
                raise AssertionError(f'Export changed source attribute {name}.')
        if output_primitive.get('material') != chunk['primitive'].get('material'):
            raise AssertionError('Export changed a primitive material assignment.')
        local_positions = accessor(check_doc, check_blob, output_primitive['attributes']['POSITION']).astype(float)
        output_matrix = (_node_matrix(check_doc['nodes'][root_index])
                         @ _node_matrix(check_doc['nodes'][node_index]))
        output_world = _transform_positions(local_positions, output_matrix)[inverse.reshape(-1, 3)]
        expected_world = chunk['tri'][chosen]
        if not np.allclose(output_world, expected_world, atol=1e-6):
            raise AssertionError('Export changed world-space triangle coordinates.')
    return path


def _source_arrays(chunks):
    return {
        'source_instance': np.concatenate([np.full(len(chunk['faces']), chunk['source_instance'], dtype=np.uint32)
                                           for chunk in chunks]),
        'source_node': np.concatenate([np.full(len(chunk['faces']), chunk['source_node'], dtype=np.uint32)
                                       for chunk in chunks]),
        'source_mesh': np.concatenate([np.full(len(chunk['faces']), chunk['source_mesh'], dtype=np.uint32)
                                       for chunk in chunks]),
        'source_primitive': np.concatenate([np.full(len(chunk['faces']), chunk['source_primitive'], dtype=np.uint32)
                                            for chunk in chunks]),
        'source_face': np.concatenate([np.arange(len(chunk['faces']), dtype=np.uint32)
                                       for chunk in chunks]),
    }


def _read_json_if_present(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def _remove_generated_file(output, relative):
    output = output.resolve()
    target = (output / relative).resolve()
    try:
        target.relative_to(output)
    except ValueError:
        return
    if target.is_file() or target.is_symlink():
        target.unlink()


def _clean_generated_output(output):
    previous_manifest = _read_json_if_present(output / 'manifest.json')
    previous_scene = _read_json_if_present(output / 'collision-scene.json')
    previous_report = _read_json_if_present(output / 'run-report.json')
    generated = {
        'segmented-scene.glb', 'face-labels.npz', 'manifest.json', 'inventory.json',
        'collision-scene.json', 'run-report.json', 'annotations-used.json',
        'review-used.json', 'corrections-used.json', 'labels-used.json', 'source-license.json',
    }
    generated.update(
        obj.get('asset') for obj in previous_manifest.get('objects', [])
        if isinstance(obj, dict) and isinstance(obj.get('asset'), str))
    for obj in previous_scene.get('objects', []):
        if isinstance(obj, dict):
            generated.update(path for path in (obj.get('npz'), obj.get('obj'))
                             if isinstance(path, str))
    generated.update(path for path in previous_report.get('previews', [])
                     if isinstance(path, str))
    generated.update({
        'previews/scene-labeled.png', 'previews/top-labeled.png',
        'previews/scene-textured.png', 'previews/annotation-map.png',
        'previews/objects-preview.png',
    })
    for relative in generated:
        _remove_generated_file(output, relative)
    for folder_name, suffixes in (('colliders', {'.npz', '.obj'}),
                                  ('previews', {'.png'})):
        folder = output / folder_name
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in suffixes:
                path.unlink()
        try:
            folder.rmdir()
        except OSError:
            pass


def segment(input_glb, output_directory, annotations=None, review=None, corrections=None, labels=None,
            inventory=False):
    """Label and export ``input_glb`` into ``output_directory``."""
    if inventory and any(value is not None for value in (annotations, review, corrections, labels)):
        raise ValueError('Inventory runs separately from annotations, labels, corrections and review.')
    source = Path(input_glb)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cfg = None
    if annotations:
        cfg = json.loads(Path(annotations).read_text())
        _annotation_source_check(cfg, source_hash)
        if cfg.get('reviewed_faces'):
            raise ValueError('Reviewed faces belong in a separate review file; pass it with --review.')
    correction_path = Path(corrections) if corrections else None
    label_path = Path(labels) if labels else None
    correction_cfg = None
    label_cfg = None
    if correction_path and label_path:
        raise ValueError('Keep imported labels and automatic-label corrections in separate runs.')
    if label_path:
        if cfg is not None:
            raise ValueError('Imported source-face labels cannot be combined with --annotations.')
        label_cfg = json.loads(label_path.read_text())
        _label_source_check(label_cfg, source_hash)
    if correction_path:
        if cfg is not None:
            raise ValueError('Manual corrections can be applied only to automatic labels; omit --annotations.')
        correction_cfg = json.loads(correction_path.read_text())
        _correction_source_check(correction_cfg, source_hash)
    review_cfg = None
    review_path = Path(review) if review else None
    if review:
        if correction_cfg is not None or label_cfg is not None:
            raise ValueError('Keep corrections or imported labels separate from review-only evaluation.')
        review_cfg = json.loads(review_path.read_text())
        _review_source_check(review_cfg, source_hash)
    doc, blob, chunks = load_geometry(source)
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    _clean_generated_output(out)
    if inventory:
        from object_inventory import inventory_labels
        labels, confidence, review, objects, label_metadata = inventory_labels(chunks, source_hash)
    elif label_cfg is not None:
        labels, confidence, review, objects, label_metadata = _import_face_labels(chunks, label_cfg)
        objects[:] = [obj for obj in objects if np.any(labels == int(obj['id']))]
        label_metadata['labels_source_sha256'] = source_hash
    elif cfg is None:
        labels, confidence, review, objects, label_metadata = automatic_labels(chunks)
    else:
        labels, confidence, review, objects, label_metadata = _annotation_labels(chunks, cfg)
    if correction_cfg is not None:
        count = _apply_manual_corrections(chunks, labels, objects, correction_cfg)
        objects[:] = [obj for obj in objects if np.any(labels == int(obj['id']))]
        label_metadata['manual_correction_count'] = count
        label_metadata['manual_corrections_source_sha256'] = source_hash
    cursor = 0
    for chunk in chunks:
        count = len(chunk['faces'])
        chunk['labels'] = labels[cursor:cursor + count]
        cursor += count
    if cursor != len(labels):
        raise AssertionError('Face label count does not match loaded geometry.')
    for object_spec in objects:
        object_spec.setdefault('color', PALETTE[(int(object_spec['id']) - 1) % len(PALETTE)])
        object_spec.setdefault('status', 'Provisional label')
    export_metadata = {
        'method': label_metadata['mode'],
        'scale': 'unverified source units',
        'labelValidation': 'review-only evaluation requested' if review_cfg else 'provisional',
    }
    _export_labeled_glb(out / 'segmented-scene.glb', doc, blob, chunks, objects, export_metadata)
    triangles, centers, normals, area = _all_face_arrays(chunks)
    entries = []
    for object_spec in objects:
        object_id = int(object_spec['id'])
        mask = labels == object_id
        selected = triangles[mask].reshape(-1, 3)
        entries.append({k: object_spec[k] for k in ['id', 'name', 'semantic_class', 'status', 'color']} | {
            'asset': 'segmented-scene.glb' if inventory else object_spec['name'] + '.glb',
            'triangles': int(mask.sum()),
            'surface_area_source_units_squared': float(area[mask].sum()),
            'bounds_world': [selected.min(0).tolist(), selected.max(0).tolist()] if len(selected) else None,
            'review_faces': int((mask & review).sum()),
            'confidence_mean': float(confidence[mask].mean()) if mask.any() else None,
            'confidence_min': float(confidence[mask].min()) if mask.any() else None,
        })
        if inventory:
            entries[-1]['stable_id'] = object_spec['stable_id']
        else:
            _export_labeled_glb(out / (object_spec['name'] + '.glb'), doc, blob, chunks, [object_spec], export_metadata)
    source_arrays = _source_arrays(chunks)
    source_keys = np.column_stack([source_arrays['source_instance'], source_arrays['source_face']])
    if len(np.unique(source_keys, axis=0)) != len(labels):
        raise AssertionError('Source face identifiers are not unique.')
    np.savez_compressed(
        out / 'face-labels.npz', object_id=labels.astype(np.uint32),
        label_confidence=confidence.astype(np.float32),
        boundary_review=review.astype(np.uint8), **source_arrays)
    evaluation = _evaluate_reviewed_faces(
        review_cfg, review_path, chunks, labels, objects, label_metadata['mode'])
    manifest = {
        'source_file': str(source.resolve()),
        'source_sha256': source_hash,
        'method': label_metadata['mode'],
        'validated_semantic_accuracy': bool(evaluation['valid_automatic_review']),
        'coordinate_frame': 'Input GLB world coordinates; estimated up is reported below',
        'meters_per_source_unit': None,
        'scale_status': 'Physical scale is unknown; values remain in source GLB units',
        'orientation': label_metadata.get('ground_plane'),
        'labeling': label_metadata,
        'source_geometry': {
            'nodes_with_triangles': len({chunk['source_node'] for chunk in chunks}),
            'mesh_occurrences': len(chunks),
            'triangles': int(len(labels)),
            'bounds_world': [triangles.min(axis=(0, 1)).tolist(), triangles.max(axis=(0, 1)).tolist()],
            'primitive_layout': [
                {'source_instance': chunk['source_instance'], 'source_node': chunk['source_node'],
                 'source_mesh': chunk['source_mesh'], 'source_primitive': chunk['source_primitive'],
                 'triangles': int(len(chunk['faces']))}
                for chunk in chunks
            ],
        },
        'objects': entries,
        'evaluation': evaluation,
        'validation': {
            'all_source_faces_partitioned_once': int(sum(entry['triangles'] for entry in entries)) == len(labels),
            'source_face_identifiers_unique': True,
            'original_vertex_attributes_exact': True,
            'original_materials_and_embedded_images_exact': True,
            'world_space_triangle_coordinates_exact': True,
            'source_transforms_preserved_for_output_roots': True,
        },
        'limitations': [
            'Automatic labels are geometry proposals and are not trained semantic inference.',
            'Unknown and low-confidence faces require review before operational use.',
            'Collision surfaces retain scan holes and incomplete backs.',
            'Ground labels do not establish traversability or physical scale.',
            'No Unity, Magic Leap 2, robot or navigation integration test has run.',
        ],
    }
    if inventory:
        from object_inventory import LIMITATIONS, write_inventory
        manifest['coordinate_frame'] = 'Input GLB world coordinates; source axes and units unchanged'
        manifest['limitations'] = LIMITATIONS
        manifest['inventory'] = 'inventory.json'
        write_inventory(out, manifest, chunks, labels)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_glb')
    parser.add_argument('output_directory')
    parser.add_argument('--annotations')
    parser.add_argument('--review')
    parser.add_argument('--corrections')
    parser.add_argument('--labels')
    args = parser.parse_args()
    result = segment(args.input_glb, args.output_directory, args.annotations, args.review,
                     args.corrections, args.labels)
    for obj in result['objects']:
        print(f"{obj['name']}: {obj['triangles']:,} triangles")
