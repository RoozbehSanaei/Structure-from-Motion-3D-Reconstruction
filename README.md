# TempleRing SfM — Python + C++ (no OpenCV)

## Benchmarks

- `docs/benchmark_report.md` — Consolidated Python vs C++ benchmark results vs Middlebury ground truth.
- `out/bench/` — Raw evaluator logs, summary tables, and filtered keyframe CSVs used for the benchmarks.
Two end-to-end implementations of a classic structure-from-motion (SfM) pipeline for the TempleRing sequence:

- `python/` — Python implementation (KLT tracking + local bundle adjustment + loop closure + pose graph).
- `cpp/` — C++20 implementation without OpenCV (minimal linear algebra + PGM I/O).

## Quick start

## Data

This bundle includes the Middlebury **TempleRing** dataset (images + camera parameters) under:

- `data/middlebury_templeRing/templeRing/` (47 PNG images, `templeR_par.txt`, `templeR_ang.txt`, `README.txt`)
- `data/middlebury_templeRing.zip` (original download)

The camera-parameter file `templeR_par.txt` is the ground truth used by the ATE evaluation utilities.


### Python
```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/templering_sfm.py --help
```

### C++ (no OpenCV)
```bash
cd cpp
cmake -S . -B build
cmake --build build -j
```

## Metrics tools (C++)
- Ground-truth relative edge:
```bash
./build/gt_keyframe_edge --help
```

- ATE for two keyframes:
```bash
./build/ate_two_frames --help
```

## Documentation
- Algorithm overview: `docs/algorithm_overview.md`
- Language feature notes: `docs/features_cpp.md`, `docs/features_python.md`

## Samples and test data
- Sample artifacts: `samples/`
- Short test bundle: `data/templering_test_short.zip`

## File index (every file)

### Root
- `LICENSE` — License for the combined repository (MIT).
- `README.md` — Top-level overview, build/run pointers, and complete file index for this repository.

### data/
- `data/templering_test_short.zip` — Short TempleRing test bundle used for reproducing the example run (images + metadata).

### docs/
- `docs/algorithm_overview.md` — Plain-language algorithm overview (technical terms defined).
- `docs/features_cpp.md` — Explanation of C++ language/library features used by the C++ implementation.
- `docs/features_python.md` — Explanation of Python language features used by the Python implementation.
- `docs/README.md` — Index of documentation files under `docs/`.

### python/
- `python/README.md` — Python-specific setup and run instructions.
- `python/requirements.txt` — Python dependencies for the pipeline.
- `python/src/templering_sfm.py` — Main runnable Python pipeline (KLT tracking + local BA + loop closure + pose-graph).

### cpp/
- `cpp/CMakeLists.txt` — CMake build configuration for the C++ implementation.
- `cpp/include/linalg.hpp` — Minimal vector/matrix math, SO(3) exp/log, and small utilities used across the C++ pipeline.
- `cpp/include/dense.hpp` — Small dense linear algebra helpers (Gaussian elimination, 3×3 inverse) used by local BA and pose-graph.
- `cpp/include/so3.hpp` — Additional SO(3) helpers (hat/vee and rotation-vector extraction) used for edge export.
- `cpp/include/pgm_io.hpp` — PGM/PPM image I/O utilities for the C++ pipeline (no external deps).
- `cpp/LICENSE` — License text carried from the C++ sub-repository.
- `cpp/README.md` — C++-specific build and run instructions.
- `cpp/README_REPO.md` — Additional repository notes for the C++ no-OpenCV implementation.
- `cpp/src/templering_sfm.cpp` — Main C++20 pipeline implementation (no OpenCV).
- `cpp/tools/ate_two_frames.cpp` — Tool: compute ATE (RMSE) for two keyframes by aligning estimated centers to ground truth from Middlebury *_par.txt.
- `cpp/tools/ate_keyframes.cpp` — Tool: compute ATE (RMSE) over N keyframes (e.g., 4) using Umeyama alignment (Sim(3) or SE(3)) against Middlebury *_par.txt ground truth.
- `cpp/tools/convert_templering_png_to_pgm.py` — Helper tool: convert TempleRing PNG images to binary PGM for the C++ pipeline.
- `cpp/tools/gt_keyframe_edge.cpp` — Tool: compute a ground-truth relative-pose edge between two keyframes from Middlebury *_par.txt.

### samples/
- `samples/cpp/camera_trajectory.png` — Rendered view of estimated camera centers / trajectory.
- `samples/cpp/inlier_matches.png` — Visualization of inlier correspondences after RANSAC (two-view geometry).
- `samples/cpp/input_montage.png` — Input montage image (subset of TempleRing frames used in a short test).
- `samples/cpp/keyframes_camera_centers.csv` — Keyframe camera centers export (CSV).
- `samples/cpp/posegraph_edges.csv` — Pose-graph edge constraints (CSV) from loop-closure / pose-graph stage.
- `samples/cpp/sparse_pointcloud.png` — Rendered view of the reconstructed sparse 3D point cloud.
- `samples/cpp/templeRing_sparse_points.ply` — Sparse 3D point cloud export (PLY format).
- `samples/python/camera_trajectory.png` — Rendered view of estimated camera centers / trajectory.
- `samples/python/inlier_matches.png` — Visualization of inlier correspondences after RANSAC (two-view geometry).
- `samples/python/input_montage.png` — Input montage image (subset of TempleRing frames used in a short test).
- `samples/python/keyframes_camera_centers.csv` — Keyframe camera centers export (CSV).
- `samples/python/posegraph_edges.csv` — Pose-graph edge constraints (CSV) from loop-closure / pose-graph stage.
- `samples/python/sparse_pointcloud.png` — Rendered view of the reconstructed sparse 3D point cloud.
- `samples/python/templeRing_sparse_points.ply` — Sparse 3D point cloud export (PLY format).
- `samples/README.md` — Explains the sample input/output artifacts stored under `samples/`.
