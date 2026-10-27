# Handoff

Updated 2026-10-15. Read `AGENTS.md`, `agent.md`, then this file.

## Goal

Build a Python-first **offline GLB-to-3D-object workflow** for indoor and outdoor scenes. The desired result is one record per physical object with an ID, a supported semantic label or `unknown`, XYZ position, 3D bounds, source-face references and collision geometry in a shared scan coordinate frame. A 2D map is not a deliverable, and real-time processing is not required now. LiDAR/depth input may be considered later; GLB is the current supported format. Physical scale and scan-to-robot registration are required before coordinates can guide a robot.

## Current implementation

- `segment_objects.py` loads triangle GLBs across roots, nested transforms, meshes and primitives. It retains source instance/face IDs, UVs, textures, local attributes and world-space positions when exporting labeled 3D assets.
- `object_inventory.py`, selected by `run.py --inventory`, creates source-linked 3D surface candidates using exact shared world-space edges across all chunks. Disconnected surfaces within chunks are split; source mesh chunks are not treated as objects. Every semantic label stays `unknown`, and all physical boundaries are unverified. The inventory records stable source/membership hashes, observed bounds centers as XYZ, 3D bounds, source frame/units, explicit uncertainty and inclusive source-face ranges backed by `face-labels.npz`.
- `run.py` is the entry point for the Bucha annotation regression, generic outdoor geometry proposals, or hash-guarded imported source-face labels. It exports GLBs, a manifest, face labels, collision surfaces and a run report. `collision.py` performs static ray queries against the exported surfaces.
- `render_views.py` and `infer_labels.py` form an optional *internal* image-model proposal path. They render textured views with face IDs and vote provisional classes back onto 3D faces. `run.py --labels` imports those proposals while leaving every other face `unknown`. This is not a validated 3D object segmenter.
- `data/TEEsavR23oF.glb` is a furnished indoor living room. Its source mesh occurrences may be reconstruction chunks, not physical objects. The old building/ground/rubble labels are out of domain for it. The previous five-view model run labeled only 883 of 395,018 faces, without human semantic validation; no sofa or chair was established.
- `data/destroyed_apartments_bucha_ukraine_war_2022.glb` and `annotations.json` remain as a scan-specific outdoor regression. The default `python run.py` runs that regression; it is not evidence of general scene understanding.

## Cleanup on 2026-10-15

The user removed the need for a 2D map. `project_map.py`, `project_labels.py`, the map CLI flags and map-only tests were removed. All ignored `results*` directories, including 3.2 GB of living-room smoke outputs, were deleted because they are generated artifacts. Both input GLBs, `.venv/`, `models/`, the internal label-proposal path, 3D exports and collision code were retained. Generated outputs must be rerun when needed. The repo had substantial pre-existing uncommitted changes; do not discard them while continuing work.

## Verification

`.venv/bin/python -m unittest discover -s tests -v` passed **18 tests** after inventory implementation (4.6 seconds). The four new tests cover connectivity across and within chunks, exact versus point/near contacts, degenerate/nonmanifold faces, stable IDs, bounds, face ranges, unchanged source bytes, texture attributes, collision queries and stale-output cleanup. Tests use temporary files. The unchanged default Bucha regression also completed with `.venv/bin/python run.py --no-previews` in 4.2 seconds. No packages were downloaded.

## Run the inventory

```sh
.venv/bin/python run.py --inventory --input data/TEEsavR23oF.glb \
  --output results-living-room-inventory
```

Verified 2026-10-15 in **8.6 seconds**: **395,018 faces, 209 source occurrences, 159 candidates**. The largest candidate contains **381,667 faces (96.62%) across 186 source occurrences**, with observed bounds center `[-4.158688, 4.409025, 2.816410]`. It spans the scene bounds and is unresolved structure, not one physical object. **134 candidates have ten faces or fewer** and may be fragments. There are 48 candidates joining multiple source occurrences, while 67 source occurrences split across candidates. All labels are `unknown`; no sofa, chair or other semantic object has been established, and zero physical objects are verified.

Outputs are retained under ignored `results-living-room-inventory/` (about 63 MB): `inventory.json`, `face-labels.npz`, one combined `segmented-scene.glb`, collision files, manifest and run report. Individual visual GLBs are deliberately not duplicated in this mode; `objectId` extras identify each group in the combined scene. Existing collision query commands work with this folder's `collision-scene.json`. Inventory mode always skips images, does not run the outdoor heuristics, and runs separately from annotations, labels, corrections and review. Reusing an output directory removes stale inventory data along with other generated files.

Every real-scan inventory range, bound and collision triangle was independently checked against the loader's source geometry; all 395,018 source faces occur exactly once. Export checks confirm textures, UV/local attributes and world transforms are preserved, and the derived collision ray returned the expected candidate and source face. This verifies structure, not semantic or physical-instance accuracy.

## Verified limitations and remaining work

Shared-edge connectivity cannot separate touching furniture, floors and walls, and reconstruction bridges may join unrelated surfaces. Holes, disconnected object parts, unequal edge tessellation and tiny transform rounding differences may fragment one object. No tolerance, gap closing, semantic classifier or name-based labeling is used. Zero-area triangles are retained as separate candidates. A candidate is never asserted to be a complete or verified physical object.

XYZ is the observed axis-aligned bounds center in **source GLB world units and axes**, with all nested node transforms applied. It is not a calibrated object origin or robot pose. Meters per source unit, position error and scan-to-robot registration remain unknown. IDs are stable for identical source bytes and source-face membership; they do not identify objects across edited GLBs. Keep the next work focused on reviewed physical boundaries and supported semantics before claiming object recognition. No additional feature is authorized by this handoff alone.

## Constraints

Follow `agent.md`: no subagents, no overengineering, no edits for question-only requests, and pin any downloaded packages to versions at least two weeks old. Preserve source hashes, world-coordinate consistency and original geometry attributes. Do not claim source units are meters or that a collision surface implies traversability. The Bucha source has license evidence in `source-license.json`; verify its acquisition terms before redistributing it publicly. The local repo has no configured remote.

## Single-photo depth prototype — 2026-10-26

`image_depth.py` accepts a local image and runs the pinned ONNX Community Depth Anything V2 Small relative model using the existing ONNX Runtime, NumPy and Pillow. `download_depth_model.py` verifies the pinned revision and SHA-256. See README for the command, model provenance, preprocessing and interpretation. The raw `.npy` contains the model-resolution output, where larger is relatively nearer; the grayscale visualization is separately normalized and resized. The original photo remains read only, output is restricted to a new `results-*` directory, and no GLB face mapping is claimed.

The checkpoint was downloaded and verified; local inference ran on a synthetic solid-color image only. A user photograph has not yet been supplied or processed. The model cannot infer metric scale or unseen surfaces, and its results do not establish object boundaries or reproduce GATOR. No packages were installed. Existing GLB workflows were left unchanged.

Verification on 2026-10-26: all 21 unit tests passed, including three depth preprocessing/output checks. The default Bucha regression completed with `--no-previews` in a separate output directory; the living-room inventory also completed with 159 unknown candidates. Both are structural regressions, not semantic validation. Synthetic inference produced a finite float32 array of shape 294 × 518 for a 640 × 360 input.

On 2026-10-26, the user attached a 1174 × 1462 interior image and local inference succeeded. Outputs are in ignored `results-attached-photo/`: the raw float32 model output is 518 × 420, with a separate original-size grayscale visualization and report. The foreground sofa appears nearer and the back of the room farther in the visualization; no ground-truth depth validation was performed. The attached input was a temporary file, so its original path may not persist.

A later requested experimental single-image relief preview was generated in ignored `results-photo-surface/` from the attached image and relative depth. It includes a vertex-colored PLY, oblique PNG, generation script and notes documenting arbitrary depth and camera assumptions. This is not measured 3D geometry or collision data; occluded surfaces are missing. The script depends on the original temporary screenshot path recorded in the depth report.

## Local depth web demo

The user requested an image-upload demo with depth comparison and a rotatable relief; object labels are deferred. Run `.venv/bin/python depth_demo_server.py` and open `http://127.0.0.1:8765`. The localhost Python server uses the existing verified ONNX checkpoint, processes uploaded images in memory, and serves `depth_demo/`. The comparison offers a downloadable depth PNG. The WebGL relief uses an arbitrary relative-depth mapping and assumed 60° field of view; it is not measured geometry. No new packages, external hosting, or changes to the GLB workflow were introduced.

## Metric depth demo

The local demo supports an indoor/outdoor metric model selection before image upload and click/keyboard sampling of estimated camera-axis depth in meters. Run `.venv/bin/python download_metric_models.py` once and restart the server. Indoor and outdoor ONNX exports are pinned and checksum-verified; the existing relative model still supplies the comparison and 3D relief. See README for source revisions, licenses, preprocessing, output semantics and limitations. Fixed 518 × 518 model inputs require stretching the image; metric output is resampled to 256 × 256 for click lookup. No per-camera calibration, true-scale validation, or range accuracy is established. On the user's outdoor image both metric checkpoints produced finite output, but their numbers differed significantly. The outdoor export lacks a model card; some official model weights were independently matched, but numerical equivalence has not been verified.

## Live camera benchmark

`live_depth.py` is a separate local OpenCV camera preview and timed benchmark, leaving the website intact. It selects the existing indoor/outdoor metric or relative ONNX model, displays camera/depth side by side, and reports end-to-end FPS and inference time after warm-up. Frames are not saved. The optional `requirements-live.txt` pins OpenCV 4.11.0.86, released 2025-01-16; installed with `--no-deps`. 27 tests passed, including live preprocessing and depth polarity. A five-frame repeated still-image check on the indoor model took 0.533 seconds per frame for preprocessing, inference and depth-image preparation, excluding capture and window display. This is not a live benchmark. On 2026-10-26 the camera attempt failed with OpenCV `not authorized to capture video (status 0)` and no frames processed; the user needs to grant macOS camera access before an actual camera benchmark can be reported.

After the user enabled camera access, `live_depth.py --mode indoor --seconds 15` ran successfully on 2026-10-26: 26 camera frames in 15.30 seconds, 1.70 end-to-end FPS, mean inference 520.4 ms (1.92 inference FPS), after two warm-up frames. Camera opening initially returned an empty frame; the script now retries for up to eight seconds during warm-up. Three live-processing unit tests passed after the change. Frames were not saved.

The user requested a simpler 1990s appearance for the local website. The upload, depth comparison, metric estimate and 3D controls remain in `depth_demo/`; the interface now uses a Windows 95-inspired layout. The separate live camera prototype remains available and unchanged. The local endpoint and demo tests passed after the visual update.

The user then requested much less CSS. The website now uses plain browser-style presentation with only the layout rules needed for comparison, the 3D canvas and click selection. Demo tests and JavaScript syntax check pass.

On 2026-10-26 the user asked to keep the website simple and reduce CSS. The page styling and heading/footer markup were simplified without removing upload, relative depth, 3D rotation or metric selection. All 27 tests passed; a synthetic local HTTP request returned relative depth, metric depth and surface data. The page was visually checked in a browser.
