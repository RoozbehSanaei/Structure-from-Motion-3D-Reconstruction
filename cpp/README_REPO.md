# TempleRing classic SfM (C++20, no OpenCV)

This repository contains a self-contained C++20 implementation of a classic SfM pipeline for the TempleRing sequence **without OpenCV**.

## What you get
- KLT-style tracking (pyramidal Lucas–Kanade) implemented directly
- Essential matrix estimation (8-point + RANSAC) implemented directly
- Triangulation (DLT) implemented directly
- Exports:
  - `templeRing_sparse_points.ply`
  - `keyframes_camera_centers.csv`

## Repository layout
- `src/` pipeline implementation
- `include/` minimal linear algebra + PGM/PPM I/O
- `tools/` dataset conversion helper (PNG → PGM)

## Build
```bash
cmake -S . -B build
cmake --build build -j
```

## Prepare images (PNG → PGM)
```bash
python3 tools/convert_templering_png_to_pgm.py <templering_root>
```

## Run
```bash
./build/templering_sfm <templering_root> <out_dir> [frames]
```
