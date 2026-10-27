"""Fuse local image semantic predictions onto visible GLB source faces.

This uses a small ADE20K SegFormer ONNX checkpoint. Its labels are proposals,
not validated object identities. Faces without enough consistent image evidence
stay unknown when the resulting JSON is passed to ``run.py --labels``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image


MODEL_REVISION = "312a6466f90f7057b711cf550d5659ed3a18d508"
MODEL_SHA256 = "3a89102115fe3c16230502437b894844ba50cde6f7c800f9884e87c360bcbfc9"


def _load_model(model_dir):
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        raise ValueError("Install the pinned optional ONNX Runtime packages before local inference.") from error
    model_dir = Path(model_dir)
    if hashlib.sha256((model_dir / "model.onnx").read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("ONNX model does not match the pinned checkpoint SHA-256.")
    config = json.loads((model_dir / "config.json").read_text())
    processor = json.loads((model_dir / "preprocessor_config.json").read_text())
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
        raise ValueError("Expected a single-input, single-output SegFormer ONNX model.")
    labels = {int(index): name for index, name in config["id2label"].items()}
    return session, processor, labels


def _predict_image(session, processor, rgb):
    width, height = processor["size"]["width"], processor["size"]["height"]
    resized = np.asarray(Image.fromarray(rgb).resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32)
    image = resized * float(processor["rescale_factor"])
    image = (image - np.asarray(processor["image_mean"], dtype=np.float32)) / np.asarray(
        processor["image_std"], dtype=np.float32)
    tensor = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
    logits = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    logits -= logits.max(axis=0, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    labels = probabilities.argmax(axis=0).astype(np.uint8)
    confidence = probabilities.max(axis=0).astype(np.float32)
    shape = (rgb.shape[1], rgb.shape[0])
    labels = np.asarray(Image.fromarray(labels).resize(shape, Image.Resampling.NEAREST))
    confidence = np.asarray(Image.fromarray(confidence).resize(shape, Image.Resampling.BILINEAR))
    return labels, confidence


def _slug(value):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug if slug and slug[0].isalpha() else "class_" + slug


def infer_source_faces(input_glb, views_dir, model_dir, output_json,
                       min_pixels=4, min_agreement=.70, min_confidence=.50):
    """Write hash-guarded ``labeled_faces`` for the existing import pipeline."""
    source = Path(input_glb)
    views_dir = Path(views_dir)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    views_metadata = json.loads((views_dir / "views.json").read_text())
    if views_metadata.get("source_sha256") != source_hash:
        raise ValueError("Rendered views do not match the input GLB SHA-256.")
    if min_pixels < 1 or not 0 <= min_agreement <= 1 or not 0 <= min_confidence <= 1:
        raise ValueError("Invalid inference thresholds.")

    session, processor, class_names = _load_model(model_dir)
    totals = {}
    votes = {}
    files = views_metadata.get("views", {})
    if not files:
        raise ValueError("No views are listed in views.json.")
    for entry in files.values():
        image_path = views_dir / entry["rgb"]
        arrays_path = views_dir / entry["ids"]
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        with np.load(arrays_path) as arrays:
            instances = arrays["source_instance"]
            faces = arrays["source_face"]
        if instances.shape != rgb.shape[:2] or faces.shape != rgb.shape[:2]:
            raise ValueError(f"RGB and source ID image sizes differ for {image_path.name}.")
        classes, confidence = _predict_image(session, processor, rgb)
        visible = (instances >= 0) & (faces >= 0)
        if not visible.any():
            continue
        keys, counts = np.unique(np.stack([instances[visible], faces[visible]], axis=1),
                                 axis=0, return_counts=True)
        for (instance, face), count in zip(keys, counts):
            key = (int(instance), int(face))
            totals[key] = totals.get(key, 0) + int(count)
        confident = visible & (confidence >= min_confidence)
        if not confident.any():
            continue
        triples, inverse, counts = np.unique(
            np.stack([instances[confident], faces[confident], classes[confident]], axis=1),
            axis=0, return_inverse=True, return_counts=True)
        weights = np.bincount(inverse, weights=confidence[confident], minlength=len(triples))
        for (instance, face, class_id), count, weight in zip(triples, counts, weights):
            key = (int(instance), int(face), int(class_id))
            old_count, old_weight = votes.get(key, (0, 0.0))
            votes[key] = (old_count + int(count), old_weight + float(weight))

    winners = {}
    for (instance, face, class_id), (count, weight) in votes.items():
        key = (instance, face)
        if key not in winners or weight > winners[key][2]:
            winners[key] = (class_id, count, weight)
    assignments = []
    for (instance, face), (class_id, count, weight) in sorted(winners.items()):
        agreement = count / totals[(instance, face)]
        mean_confidence = weight / count
        if count < min_pixels or agreement < min_agreement or mean_confidence < min_confidence:
            continue
        semantic_class = class_names[class_id]
        assignments.append({
            "source_instance": instance, "source_face": face,
            "semantic_class": semantic_class,
            "object_name": f"{_slug(semantic_class)}_source_{instance:04d}",
            "confidence": round(float(agreement * mean_confidence), 4),
        })
    result = {
        "source_sha256": source_hash,
        "method": "provisional_2d_semantic_votes_on_visible_source_faces",
        "model": {"repository": "optimum/segformer-b0-finetuned-ade-512-512",
                  "revision": MODEL_REVISION,
                  "model_sha256": MODEL_SHA256},
        "thresholds": {"min_pixels": min_pixels, "min_agreement": min_agreement,
                       "min_pixel_confidence": min_confidence},
        "visible_source_faces": len(totals),
        "labeled_faces": assignments,
    }
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source GLB")
    parser.add_argument("--views", type=Path, required=True, help="Directory from render_views.py")
    parser.add_argument("--model-dir", type=Path, required=True, help="Pinned local ONNX checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output label JSON")
    parser.add_argument("--min-pixels", type=int, default=4)
    parser.add_argument("--min-agreement", type=float, default=.70)
    parser.add_argument("--min-confidence", type=float, default=.50)
    args = parser.parse_args()
    result = infer_source_faces(args.input, args.views, args.model_dir, args.output,
                                args.min_pixels, args.min_agreement, args.min_confidence)
    print(f"Labeled {len(result['labeled_faces']):,} of {result['visible_source_faces']:,} visible source faces.")


if __name__ == "__main__":
    main()
