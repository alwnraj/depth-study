# Instructions for the next agent

Read `handoff.md` for project state and `README.md` for the runnable workflow.

## User's standing instructions

- Do not use subagents or delegate work to other agents.
- Do not overengineer. Prefer a small, runnable Python solution.
- Do not change files when the user is only asking a question.
- When downloading packages, install explicitly pinned versions that are at least two weeks old. Do not download the latest package versions. Verify release dates before changing dependencies.

## Project direction

Keep loading, automatic or annotation-guided labeling, exports, previews and collision queries executable through Python scripts in VS Code. The user abandoned the website prototype. Do not introduce a browser frontend, Node.js service, Unity project or large ML installation unless the user asks for that direction.

The current goal is **offline 3D object identification for robot navigation**, across indoor and outdoor scenes. Process GLB files as a batch job: separate physical objects, assign semantic names when evidence supports them, and report each object's identity, XYZ position and 3D bounds in a shared coordinate frame. Retain unknown labels when the class or object boundary is uncertain. The robot needs 3D locations and collision geometry; a 2D map is not a required output. Real-time processing is not a current requirement. LiDAR point clouds and depth scans are possible future inputs, not implemented support. Physical scale and robot-to-scan registration must be established before treating coordinates as actionable robot positions.

The existing Bucha building/ground/rubble workflow is a scan-specific regression. The earlier 2D map code was removed as unnecessary for the current goal. The current implementation does not reliably separate physical objects or validate semantic labels on arbitrary scenes. No next feature has been authorized by the handoff alone.

## Current scope

The user confirmed that `data/TEEsavR23oF.glb` is a furnished indoor living room, then clarified that the project should work directly in 3D without requiring 2D maps. The older automatic building/ground/rubble/unknown output is semantically invalid for this scan and only demonstrates loader/export structure. The scan-specific manual corrections based on an incorrect outdoor/façade interpretation were removed. The optional ONNX image model makes sparse, unvalidated source-face proposals; its rendered image views are an internal proposal technique, not a 2D deliverable. Do not apply outdoor corrections, tune Bucha thresholds, or present the model output as validated semantic recognition without representative human review.

## Implementation rules

- Preserve original face references, UVs, texture bytes, local vertex attributes and shared world coordinates when splitting geometry. Retain uncertain labels where evidence is insufficient.
- Keep the GLB loader reusable across scene roots, nested node transforms, meshes and triangle primitives. Raise a clear unsupported-feature error instead of silently dropping geometry.
- Keep inputs read only and generated files under the configured results directory. Do not commit `.venv`, caches or generated result folders.
- Use `run.py` as the main entry point, `config.json` for paths, `annotations.json` for the Bucha regression masks, a separate review JSON for evaluation, and a source-hash-guarded correction JSON for manual face edits. `--review` evaluates automatic predictions without changing them; `--corrections` changes labels and must run separately. Do not hardcode a user's absolute paths into code or silently remove the input-hash guard; annotation, review and correction files must identify their source hash.
- The generic correction seam is implemented and tested, but no scan-specific correction file is currently valid for `TEEsavR23oF.glb`. Manual corrections require a scene-appropriate taxonomy and evidence from human review.
- Distinguish provisional visual labels, collision geometry and actual engine/robot integration. Do not call source units meters, treat ground as safe to traverse, or describe the method as a trained automatic segmenter.
- Run checks appropriate to a code change. The default end-to-end check is `python run.py` followed by `python -m unittest discover -s tests -v`. Tests expect the default masks; investigate changes before updating expected counts. The synthetic GLB test validates loader/export structure, not semantic generalization.
- Reusing a result directory must remove prior generated assets, collision files, previews, labels and provenance copies before writing the new manifest. Attach `source-license.json` only when its recorded input hash matches the current source. Reject skins, animations and morph targets before export because the exporter rewrites scene indices.

## Communication

Keep updates and answers concise. Lead with what works or the next concrete action. Explain material limitations plainly. Ask for missing information only when it affects the work, and continue independent work where possible.

Keep `handoff.md` current when the implementation, verification status or agreed direction changes. The supplied asset has separate licensing evidence in `source-license.json`; do not assume the local repository is cleared for public asset redistribution.
