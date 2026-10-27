"""Run labeling, textured object export, collision export and previews from Python."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'config.json')
    parser.add_argument('--input', type=Path, help='Override the source GLB path')
    parser.add_argument('--output', type=Path, help='Override the result directory')
    parser.add_argument('--annotations', type=Path, help='Override the annotation file')
    parser.add_argument('--review', type=Path, help='Review-only face labels used to evaluate predictions')
    parser.add_argument('--corrections', type=Path,
                        help='Source-face corrections applied to automatic labels')
    parser.add_argument('--labels', type=Path,
                        help='Hash-guarded source-face labels; all other faces remain unknown')
    parser.add_argument('--automatic', action='store_true',
                        help='Ignore configured annotations and use geometry-only labels')
    parser.add_argument('--inventory', action='store_true',
                        help='Offline 3D connected-surface candidates, unknown labels, no image outputs')
    parser.add_argument('--no-previews', action='store_true')
    args = parser.parse_args()
    if args.inventory and (args.automatic or args.annotations or args.labels or args.corrections or args.review):
        parser.error('--inventory runs separately from --automatic, annotations, labels, corrections and review.')
    try:
        import numpy
        import PIL
        from segment_objects import segment
        from collision import derive_verification_ray, export_collisions
        from render_previews import generate_previews
    except ModuleNotFoundError as error:
        parser.error(f'{error}. Run setup_env.py, then select the .venv interpreter in VS Code.')
    try:
        config_path = args.config.resolve()
        config = json.loads(config_path.read_text())
        source = (args.input or config_path.parent / config['input_glb']).resolve()
        output = (args.output or config_path.parent / config['output_directory']).resolve()
        annotation_setting = args.annotations
        if annotation_setting is None and not args.automatic and not args.labels and not args.inventory:
            annotation_setting = config.get('annotations')
        annotations = ((config_path.parent / annotation_setting).resolve()
                       if annotation_setting else None)
        review_setting = args.review if args.review is not None else config.get('review')
        review = ((config_path.parent / review_setting).resolve()
                  if review_setting else None)
        correction_setting = (args.corrections if args.corrections is not None
                              else config.get('corrections'))
        corrections = ((config_path.parent / correction_setting).resolve()
                       if correction_setting else None)
        labels = args.labels.resolve() if args.labels else None
        if not source.is_file():
            raise ValueError(f'Source GLB not found: {source}')
        if annotations is not None and not annotations.is_file():
            raise ValueError(f'Annotation file not found: {annotations}')
        if review is not None and not review.is_file():
            raise ValueError(f'Review file not found: {review}')
        if corrections is not None and not corrections.is_file():
            raise ValueError(f'Correction file not found: {corrections}')
        if labels is not None and not labels.is_file():
            raise ValueError(f'Label file not found: {labels}')
        started = perf_counter()
        mode = 'annotation-guided regression' if annotations else 'automatic geometry proposals'
        if labels:
            mode = 'imported source-face labels'
        if args.inventory:
            mode = 'offline 3D surface candidates (unverified physical objects)'
        if corrections:
            mode += ' with source-face corrections'
        print(f'[1/4] Loading the scan and creating {mode}...', flush=True)
        manifest = segment(source, output, annotations, review, corrections, labels,
                           inventory=args.inventory)
        displayed = (sorted(manifest['objects'], key=lambda obj: -obj['triangles'])[:5]
                     if args.inventory else manifest['objects'])
        for obj in displayed:
            print(f"  {obj['id']}  {obj['name']}: {obj['triangles']:,} faces", flush=True)
        if args.inventory:
            print(f"  {len(manifest['objects'])} candidates total; all labels unknown. See inventory.json.", flush=True)
        print('[2/4] Exporting one collision surface per object (NPZ + OBJ)...', flush=True)
        export_collisions(source, output)
        previews = []
        if not args.inventory and not args.no_previews and config.get('generate_previews', True):
            print('[3/4] Rendering PNG previews in Python...', flush=True)
            previews = generate_previews(source, output, annotations)
        else:
            print('[3/4] Previews skipped.', flush=True)
        print('[4/4] Verifying a derived collision ray and writing the run report...', flush=True)
        verification_ray = derive_verification_ray(output / 'collision-scene.json')
        if annotations is not None:
            shutil.copyfile(annotations, output / 'annotations-used.json')
        if review is not None:
            shutil.copyfile(review, output / 'review-used.json')
        if corrections is not None:
            shutil.copyfile(corrections, output / 'corrections-used.json')
        if labels is not None:
            shutil.copyfile(labels, output / 'labels-used.json')
        license_file = root / 'source-license.json'
        license_attached = False
        if license_file.is_file():
            license_metadata = json.loads(license_file.read_text())
            license_attached = license_metadata.get('input_sha256') == manifest['source_sha256']
        if license_attached:
            shutil.copyfile(license_file, output / 'source-license.json')
        elapsed = perf_counter() - started
        report = {'input': str(source), 'source_sha256': manifest['source_sha256'],
                  'labeling_mode': manifest['method'],
                  'total_faces': manifest['source_geometry']['triangles'],
                  'source_geometry': manifest['source_geometry'],
                  'orientation': manifest['orientation'],
                  'scale_status': manifest['scale_status'],
                  'python': sys.version.split()[0], 'numpy': numpy.__version__, 'pillow': PIL.__version__,
                  'elapsed_seconds': round(elapsed, 2), 'previews': previews,
                  'source_license_attached': license_attached,
                  'verification_ray': verification_ray, 'validation': manifest['validation'],
                  'evaluation': manifest['evaluation'],
                  'manual_correction_count': manifest['labeling'].get('manual_correction_count', 0),
                  'semantic_accuracy_validated': manifest['validated_semantic_accuracy'],
                  'unity_and_ml2_executed': False}
        if args.inventory:
            report['inventory'] = {'file': 'inventory.json',
                                   'candidate_count': len(manifest['objects']),
                                   'verified_physical_objects': 0, 'known_semantic_labels': 0}
        (output / 'run-report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'\nDone in {elapsed:.1f}s. Results: {output}', flush=True)
        print('Labels are scan-specific and provisional; collision surfaces are not a navigation planner.')
    except (ValueError, OSError, KeyError, AssertionError) as error:
        parser.exit(1, f'Pipeline stopped: {error}\n')


if __name__ == '__main__':
    main()
