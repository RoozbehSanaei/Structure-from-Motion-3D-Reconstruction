# C++ implementation (no OpenCV)

This directory provides a dependency-free C++ implementation of the classic monocular SfM/SLAM pipeline used in the Python version.

It matches the **block-level** pipeline structure:

- **Track-based front-end:** long-lived KLT tracks (forward–backward filtering + periodic replenishment)
- **Relative motion:** Essential matrix (normalized 8-point) + RANSAC + cheirality selection
- **Mapping:** triangulation of tracked correspondences into a sparse 3D point set with keyframe observations
- **Local refinement:** **sliding-window Bundle Adjustment (BA)** with robust (Huber) reprojection loss
- **Loop closure:** global image descriptor for candidate search + direct LK verification + Essential matrix check
- **Global drift reduction:** **translation pose-graph** optimization over keyframe camera centers (sequential + loop edges)

Notes:

- **Monocular scale is not observable.** Trajectory comparisons usually use **Sim(3) alignment** (scale+rotation+translation) before computing ATE.

## Build

From `cpp/`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

Example (Temple Ring directory layout as used in this repo):

```bash
./build/templering_sfm <templering_root> <out_dir> [frames] [options]
```

Concrete example (after converting PNG -> PGM into `templeRing_pgm/`):

```bash
./build/templering_sfm \
  ./data/middlebury_templeRing/middlebury_templeRing \
  ./out/cpp_rerun \
  48 \
  --export-geometry both \
  --mesh-kf 0 \
  --mesh-max-points 2500
```

### Geometry export options

- `--export-geometry none`       : no `.ply` geometry written
- `--export-geometry pointcloud` : write `templeRing_sparse_points.ply`
- `--export-geometry mesh`       : write `templeRing_mesh_sparse_kf<k>.ply`
- `--export-geometry both`       : write both point cloud + mesh

Mesh parameters:

- `--mesh-kf <k>`            : keyframe index used for 2D projection (default 0)
- `--mesh-max-points <n>`    : max vertices in mesh (default 2500)
- `--mesh-grid-px <px>`      : subsampling cell size in pixels (default 4)
- `--mesh-max-edge-px <px>`  : reject triangles with very long pixel edges (default 80)

## Outputs

- `out/templeRing_sparse_points.ply` — sparse 3D points (PLY) (only if `--export-geometry pointcloud|both`)
- `out/templeRing_mesh_sparse_kf<k>.ply` — triangle mesh from sparse points (only if `--export-geometry mesh|both`)
- `out/keyframes_camera_centers.csv` — per-keyframe camera centers (+ optional angle metadata from `*_par.txt`)
- `out/posegraph_edges.csv` — pose-graph edges (sequential + loop closures), in the same schema used by evaluation tools

## Ground-truth keyframe edge (Middlebury *_par.txt)

If you have the official Middlebury `*_par.txt` file (one line per image, with `K`, `R`, `t` where `P = K [R t]`),
you can compute the **ground-truth relative pose edge** between two keyframes and (optionally) compare it to your
estimated pose-graph edge.

```bash
./build/gt_keyframe_edge \
  --par ./templeRing/templeR_par.txt \
  --keyframes ./out/keyframes_camera_centers.csv \
  --edges ./out/posegraph_edges.csv \
  --i 0 --j 1
```

## Trajectory evaluation (ATE)

These tools compare **estimated keyframe camera centers** (`keyframes_camera_centers.csv`) to **Middlebury ground truth**
from `templeR_par.txt` (camera center computed as `C = -R^T t`).

```bash
./build/ate_keyframes --par ./templeRing/templeR_par.txt --keyframes ./out/keyframes_camera_centers.csv --count 2 --se3
./build/ate_keyframes --par ./templeRing/templeR_par.txt --keyframes ./out/keyframes_camera_centers.csv --count 2 --sim3
./build/ate_keyframes --par ./templeRing/templeR_par.txt --keyframes ./out/keyframes_camera_centers.csv --count 4 --se3
./build/ate_keyframes --par ./templeRing/templeR_par.txt --keyframes ./out/keyframes_camera_centers.csv --count 4 --sim3
```
