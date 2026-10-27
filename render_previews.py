"""Generate inspectable PNGs using NumPy and Pillow; no browser or GPU needed.

The renderer derives its view basis and bounds from the loaded scan. It is an
approximate inspection view; exported GLBs retain full source texture mapping.
"""
import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from segment_objects import accessor, load_geometry

BACKGROUND = '#f2f2ed'


def _basis_from_up(up):
    up = np.asarray(up, dtype=float)
    up /= max(np.linalg.norm(up), 1e-12)
    reference = np.array([0., 0., 1.]) if abs(up[2]) < .9 else np.array([1., 0., 0.])
    right = np.cross(reference, up)
    right /= max(np.linalg.norm(right), 1e-12)
    forward = np.cross(up, right)
    forward /= max(np.linalg.norm(forward), 1e-12)
    return right, forward, up


def _scene_basis(manifest):
    labeling = manifest.get('labeling', {})
    basis = labeling.get('basis')
    if basis:
        return (np.asarray(basis['right'], dtype=float),
                np.asarray(basis['forward'], dtype=float),
                np.asarray(basis['up'], dtype=float))
    plane = labeling.get('ground_plane') or manifest.get('orientation')
    if isinstance(plane, dict) and plane.get('normal') is not None:
        return _basis_from_up(plane['normal'])
    return _basis_from_up([0, 0, 1])


def render(triangles, colors, size=(1300, 850), basis=None, window=None, shade=True):
    """Render triangles with a data-derived view basis and optional world window."""
    image = Image.new('RGB', size, BACKGROUND)
    if not len(triangles):
        return image
    if basis is None:
        right, up, direction = _basis_from_up([0, 0, 1])
    else:
        right, up, direction = [np.asarray(axis, dtype=float) for axis in basis]
        direction /= max(np.linalg.norm(direction), 1e-12)
        right /= max(np.linalg.norm(right), 1e-12)
        up /= max(np.linalg.norm(up), 1e-12)
    xy = np.stack([triangles @ right, triangles @ up], axis=-1)
    if window is None:
        lo = xy.min(axis=(0, 1))
        hi = xy.max(axis=(0, 1))
        extent = np.maximum(hi - lo, 1e-6)
        scale = min(size[0] * .90 / extent[0], size[1] * .90 / extent[1])
        xy = (xy - (lo + hi) / 2) * scale
        xy[..., 0] += size[0] / 2
        xy[..., 1] = size[1] / 2 - xy[..., 1]
    else:
        left, right_edge, bottom, top_edge = window
        width = max(right_edge - left, 1e-12)
        height = max(top_edge - bottom, 1e-12)
        xy[..., 0] = (xy[..., 0] - left) * size[0] / width
        xy[..., 1] = (top_edge - xy[..., 1]) * size[1] / height
    colors = np.asarray(colors, dtype=float)
    if shade:
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([.2, -.5, 1.])
        light /= np.linalg.norm(light)
        colors = colors * (.60 + .40 * np.abs(normals @ light))[:, None]
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    draw = ImageDraw.Draw(image)
    order = np.argsort(triangles.mean(axis=1) @ direction, kind='stable')
    polygons = xy.tolist()
    rgb = colors.tolist()
    for index in order:
        draw.polygon([tuple(point) for point in polygons[index]], fill=tuple(rgb[index]))
    return image


def _image_bytes(doc, blob, image_index):
    image = doc['images'][image_index]
    if 'bufferView' in image:
        view = doc['bufferViews'][image['bufferView']]
        start = view.get('byteOffset', 0)
        return blob[start:start + view['byteLength']]
    uri = image.get('uri', '')
    if uri.startswith('data:'):
        return base64.b64decode(uri.split(',', 1)[1])
    return None


def texture_face_colors(doc, blob, chunks):
    """Sample one texture pixel per face, with material-color fallbacks."""
    images = {}
    result = []
    for chunk in chunks:
        primitive = chunk['primitive']
        material_index = primitive.get('material')
        material = doc.get('materials', [])[material_index] if material_index is not None else {}
        pbr = material.get('pbrMetallicRoughness', {})
        base_factor = np.asarray(pbr.get('baseColorFactor', [1, 1, 1, 1]), dtype=float)[:3] * 255
        texture_info = pbr.get('baseColorTexture')
        if not texture_info or 'TEXCOORD_0' not in primitive.get('attributes', {}):
            result.append(np.tile(base_factor, (len(chunk['faces']), 1)))
            continue
        texture_index = texture_info['index']
        image_index = doc['textures'][texture_index]['source']
        if image_index not in images:
            raw = _image_bytes(doc, blob, image_index)
            if not raw:
                result.append(np.tile(base_factor, (len(chunk['faces']), 1)))
                continue
            images[image_index] = np.asarray(Image.open(BytesIO(raw)).convert('RGB'))
        pixels = images[image_index]
        uv = accessor(doc, blob, primitive['attributes']['TEXCOORD_0'], decode_normalized=True)[chunk['faces']].mean(axis=1)
        uv %= 1.0
        u = np.minimum((uv[:, 0] * pixels.shape[1]).astype(int), pixels.shape[1] - 1)
        v = np.minimum((uv[:, 1] * pixels.shape[0]).astype(int), pixels.shape[0] - 1)
        result.append(pixels[v, u].astype(float) * base_factor / 255)
    return np.concatenate(result)


def framed(image, title, objects, note):
    columns = min(4, max(1, len(objects)))
    rows = max(1, int(np.ceil(len(objects) / columns)))
    footer_height = 57 + rows * 28 + 30
    result = Image.new('RGB', (image.width, image.height + footer_height), BACKGROUND)
    result.paste(image, (0, 52))
    draw = ImageDraw.Draw(result)
    draw.text((26, 16), title, fill='#22342c', font=ImageFont.load_default(size=25))
    font = ImageFont.load_default(size=17)
    cell_width = max(1, image.width // columns)
    for index, obj in enumerate(objects):
        x = 26 + (index % columns) * cell_width
        y = image.height + 67 + (index // columns) * 28
        draw.rectangle((x, y + 2, x + 14, y + 17), fill=obj['color'])
        draw.text((x + 22, y), f"{obj['id']}: {obj['name']}", fill='#22342c', font=font)
    draw.text((26, result.height - 29), note, fill='#536359', font=ImageFont.load_default(size=15))
    return result


def _annotation_map(triangles, colors, cfg, size):
    mapping = cfg.get('polygon_mapping')
    if mapping is None:
        raise ValueError('Annotation polygons require polygon_mapping for preview rendering.')
    width, height = size
    origin = np.asarray(mapping['origin'], dtype=float)
    scale = np.asarray(mapping['pixels_per_world_unit'], dtype=float)
    x0, x1 = origin[0], origin[0] + width / scale[0]
    y0, y1 = origin[1], origin[1] + height / scale[1]
    window = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    image = render(triangles, colors, size=size,
                   basis=(np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1.])),
                   window=window, shade=False)
    draw = ImageDraw.Draw(image)
    for obj in cfg.get('objects', []):
        for polygon in obj.get('polygons_pixels', []):
            points = [tuple(point) for point in polygon + [polygon[0]]]
            draw.line(points, fill=obj.get('color', '#ffffff'), width=max(1, width // 500))
            center = np.asarray(polygon).mean(axis=0)
            draw.text(tuple(center), str(obj['id']), fill=obj.get('color', '#ffffff'),
                      font=ImageFont.load_default(size=max(12, width // 50)), stroke_width=2, stroke_fill='white')
    return image


def generate_previews(input_glb, result_directory, annotations=None):
    result_directory = Path(result_directory)
    folder = result_directory / 'previews'
    folder.mkdir(exist_ok=True)
    manifest = json.loads((result_directory / 'manifest.json').read_text())
    cfg = json.loads(Path(annotations).read_text()) if annotations else None
    objects = manifest['objects']
    doc, blob, chunks = load_geometry(input_glb)
    triangles = np.concatenate([chunk['tri'] for chunk in chunks])
    with np.load(result_directory / 'face-labels.npz') as data:
        labels = data['object_id']
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for obj in objects:
        color = tuple(int(obj['color'][index:index + 2], 16) for index in (1, 3, 5))
        colors[labels == obj['id']] = color
    right, forward, up = _scene_basis(manifest)
    perspective_direction = up + right * .45 + forward * .45
    perspective_direction /= max(np.linalg.norm(perspective_direction), 1e-12)
    note = 'Provisional labels. Estimated up is reported in manifest; source scale is unverified.'
    framed(render(triangles, colors, basis=(right, up, perspective_direction)),
           'Scan: separate object labels', objects, note).save(folder / 'scene-labeled.png')
    framed(render(triangles, colors, basis=(right, forward, up), shade=False),
           'Top-down labels: estimated ground frame', objects, note).save(folder / 'top-labeled.png')
    sampled_texture = texture_face_colors(doc, blob, chunks)
    framed(render(triangles, sampled_texture, basis=(right, up, perspective_direction), shade=False),
           'Original texture: approximate per-face preview', objects,
           'PNG uses one texture sample per face. Exported GLBs keep full source UV mapping.').save(folder / 'scene-textured.png')
    previews = ['previews/scene-labeled.png', 'previews/top-labeled.png', 'previews/scene-textured.png']
    if cfg:
        if cfg.get('preview_size'):
            preview_size = tuple(cfg['preview_size'])
        else:
            polygons = [point for obj in cfg.get('objects', []) for polygon in obj.get('polygons_pixels', [])
                        for point in polygon]
            maximum = (np.max(np.asarray(polygons, dtype=float), axis=0) + 1
                       if polygons else np.array([1, 1]))
            preview_size = tuple(np.maximum(maximum.astype(int), [1, 1]))
        annotation_image = _annotation_map(triangles, sampled_texture, cfg, preview_size)
        annotation_image.save(folder / 'annotation-map.png')
        previews.append('previews/annotation-map.png')
    columns = min(3, max(1, len(objects)))
    rows = max(1, int(np.ceil(len(objects) / columns)))
    cell_width, cell_height = 620, 390
    sheet = Image.new('RGB', (columns * cell_width, rows * cell_height + 70), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((24, 18), 'Separate assets: provisional object labels', fill='#22342c', font=ImageFont.load_default(size=25))
    for index, obj in enumerate(objects):
        selected = labels == obj['id']
        x, y = (index % columns) * cell_width, 64 + (index // columns) * cell_height
        sheet.paste(render(triangles[selected], colors[selected], size=(cell_width, 320),
                           basis=(right, up, perspective_direction)), (x, y))
        draw.text((x + 22, y + 324), obj['name'], fill='#22342c', font=ImageFont.load_default(size=22))
        draw.text((x + 22, y + 355), f"{obj['triangles']:,} faces | ID {obj['id']}",
                  fill='#536359', font=ImageFont.load_default(size=17))
    sheet.save(folder / 'objects-preview.png')
    previews.append('previews/objects-preview.png')
    return previews


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_glb')
    parser.add_argument('result_directory')
    parser.add_argument('--annotations')
    args = parser.parse_args()
    for filename in generate_previews(args.input_glb, args.result_directory, args.annotations):
        print(filename)
