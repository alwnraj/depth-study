"""Render six orthographic textured views with source face IDs.

This small CPU rasterizer is intended to create inputs for image segmentation
models. It is not a physically based renderer and only handles opaque triangle
primitives loaded by :func:`segment_objects.load_geometry`.
"""
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from segment_objects import accessor, load_geometry


VIEWS = {
    "+X": ([1, 0, 0], [0, 0, 1]), "-X": ([-1, 0, 0], [0, 0, 1]),
    "+Y": ([0, 1, 0], [0, 0, 1]), "-Y": ([0, -1, 0], [0, 0, 1]),
    "+Z": ([0, 0, 1], [0, 1, 0]), "-Z": ([0, 0, -1], [0, 1, 0]),
}


def _embedded_image(doc, blob, image_index):
    image = doc["images"][image_index]
    if "bufferView" not in image:
        raise ValueError("Only embedded bufferView images are supported.")
    view = doc["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    end = start + view["byteLength"]
    with Image.open(BytesIO(blob[start:end])) as decoded:
        return np.asarray(decoded.convert("RGB"))


def _materials(doc, blob):
    cache = {}
    result = []
    for material in doc.get("materials", []):
        if material.get("alphaMode", "OPAQUE") != "OPAQUE":
            raise ValueError("Semantic view rendering does not support transparent glTF materials.")
        pbr = material.get("pbrMetallicRoughness", {})
        factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1])[:3], dtype=float)
        info = pbr.get("baseColorTexture")
        texture = None
        texcoord = 0
        if info is not None:
            if "KHR_texture_transform" in info.get("extensions", {}):
                raise ValueError("Semantic view rendering does not support KHR_texture_transform.")
            texcoord = info.get("texCoord", 0)
            texture_index = info["index"]
            image_index = doc["textures"][texture_index]["source"]
            if image_index not in cache:
                cache[image_index] = _embedded_image(doc, blob, image_index)
            texture = cache[image_index]
        result.append((factor, texture, texcoord))
    return result


def render_view(path, direction, resolution=512, background=(20, 22, 26), exclude_names=None):
    """Return ``(rgb, source_instance, source_face)`` for one orthographic view.

    ``direction`` is one of ``+X``, ``-X``, ``+Y``, ``-Y``, ``+Z`` or ``-Z``.
    Pixel ID arrays are int32 and use -1 for background. Resolution may be an
    integer or ``(width, height)``. Geometry is fitted to the image with a 3%
    border and all faces are treated as double-sided.
    """
    if direction not in VIEWS:
        raise ValueError(f"Unknown view {direction!r}; expected one of {tuple(VIEWS)}.")
    if isinstance(resolution, int):
        width = height = resolution
    else:
        width, height = map(int, resolution)
    if width < 1 or height < 1:
        raise ValueError("Resolution dimensions must be positive.")

    doc, blob, chunks = load_geometry(path)
    excluded = {str(name).casefold() for name in (exclude_names or [])}
    if excluded:
        chunks = [chunk for chunk in chunks if not any(
            term in " ".join((doc["nodes"][chunk["source_node"]].get("name", ""),
                               doc["meshes"][chunk["source_mesh"]].get("name", ""))).casefold()
            for term in excluded)]
    if not chunks:
        raise ValueError("No geometry remains after applying exclude_names.")
    view_dir, up = (np.asarray(v, dtype=float) for v in VIEWS[direction])
    right = np.cross(view_dir, up)
    all_points = np.concatenate([chunk["tri"].reshape(-1, 3) for chunk in chunks])
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    projected = np.column_stack(((all_points - center) @ right, (all_points - center) @ up))
    span = np.maximum(projected.max(axis=0) - projected.min(axis=0), 1e-8)
    scale = min((width - 1) * .94 / span[0], (height - 1) * .94 / span[1])
    screen_all = np.column_stack((width / 2 + projected[:, 0] * scale,
                                  height / 2 - projected[:, 1] * scale))

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:] = background
    source_instance = np.full((height, width), -1, dtype=np.int32)
    source_face = np.full((height, width), -1, dtype=np.int32)
    depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
    materials = _materials(doc, blob)

    point_offset = 0
    for chunk in chunks:
        tri = chunk["tri"]
        count = len(tri)
        screen = screen_all[point_offset:point_offset + count * 3].reshape(count, 3, 2)
        point_offset += count * 3
        depths = tri @ view_dir
        prim = chunk["primitive"]
        attrs = prim.get("attributes", {})
        uv = None
        material_index = prim.get("material")
        material = materials[material_index] if material_index is not None and material_index < len(materials) else (np.ones(3), None, 0)
        if material[1] is not None:
            attr_name = f"TEXCOORD_{material[2]}"
            if attr_name not in attrs:
                raise ValueError(f"Textured primitive is missing {attr_name}.")
            uv = accessor(doc, blob, attrs[attr_name], decode_normalized=True)[chunk["faces"]]

        for face_index, vertices in enumerate(screen):
            min_x = max(0, int(np.floor(vertices[:, 0].min())))
            max_x = min(width - 1, int(np.ceil(vertices[:, 0].max())))
            min_y = max(0, int(np.floor(vertices[:, 1].min())))
            max_y = min(height - 1, int(np.ceil(vertices[:, 1].max())))
            if min_x > max_x or min_y > max_y:
                continue
            ax, ay = vertices[0]
            bx, by = vertices[1]
            cx, cy = vertices[2]
            denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denom) < 1e-12:
                continue
            ys, xs = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
            px, py = xs + .5, ys + .5
            w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
            face_depth = w0 * depths[face_index, 0] + w1 * depths[face_index, 1] + w2 * depths[face_index, 2]
            visible = inside & (face_depth >= depth_buffer[min_y:max_y + 1, min_x:max_x + 1])
            if not visible.any():
                continue
            target_depth = depth_buffer[min_y:max_y + 1, min_x:max_x + 1]
            target_depth[visible] = face_depth[visible]
            target_inst = source_instance[min_y:max_y + 1, min_x:max_x + 1]
            target_face = source_face[min_y:max_y + 1, min_x:max_x + 1]
            target_inst[visible] = chunk["source_instance"]
            target_face[visible] = face_index

            color = np.broadcast_to(material[0], (*visible.shape, 3)).copy()
            if uv is not None:
                face_uv = uv[face_index]
                interpolated = (w0[..., None] * face_uv[0] + w1[..., None] * face_uv[1]
                                + w2[..., None] * face_uv[2])
                texture = material[1]
                tx = np.clip(np.rint(interpolated[..., 0] * (texture.shape[1] - 1)).astype(int), 0, texture.shape[1] - 1)
                # glTF UV v=0 addresses the bottom of the texture; decoded image
                # arrays use row 0 for the top.
                ty = np.clip(np.rint((1 - interpolated[..., 1]) * (texture.shape[0] - 1)).astype(int), 0, texture.shape[0] - 1)
                color *= texture[ty, tx] / 255.
            color = np.clip(np.rint(color * 255), 0, 255).astype(np.uint8)
            rgb_region = rgb[min_y:max_y + 1, min_x:max_x + 1]
            rgb_region[visible] = color[visible]
    return rgb, source_instance, source_face


def render_views(path, output_dir, resolution=512, exclude_names=None, directions=None):
    """Save RGB PNGs and paired ID arrays for selected canonical views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = tuple(VIEWS if directions is None else directions)
    if not directions or any(direction not in VIEWS for direction in directions):
        raise ValueError(f"Directions must be chosen from {tuple(VIEWS)}.")
    for direction in directions:
        rgb, instances, faces = render_view(path, direction, resolution, exclude_names=exclude_names)
        stem = direction.replace("+", "p").replace("-", "m").lower()
        Image.fromarray(rgb).save(output_dir / f"view_{stem}.png")
        np.savez_compressed(output_dir / f"view_{stem}.npz",
                            source_instance=instances, source_face=faces)
    metadata = {
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "views": {
            direction: {"rgb": f"view_{direction.replace('+', 'p').replace('-', 'm').lower()}.png",
                        "ids": f"view_{direction.replace('+', 'p').replace('-', 'm').lower()}.npz"}
            for direction in directions
        },
    }
    (output_dir / "views.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--direction", action="append", choices=tuple(VIEWS),
                        help="Repeat to render a subset; default is all six")
    parser.add_argument("--exclude-name", action="append", default=[],
                        help="Exclude source node or mesh names containing this text")
    args = parser.parse_args()
    render_views(args.input, args.output, args.resolution, args.exclude_name, args.direction)
    print(f"Rendered {len(args.direction or VIEWS)} views in {args.output}.")


if __name__ == "__main__":
    main()
