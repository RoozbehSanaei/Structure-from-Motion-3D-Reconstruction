# Benchmark report — Python vs C++ (TempleRing, ground truth)

This report consolidates the **ground-truth benchmark results** for the Python and C++ implementations using the **existing sample outputs** in this repository (no pipeline rerun required).

## Scope

Metrics included (as discussed):
- **Keyframe edge error (0→1)** vs ground truth:
  - rotation error (degrees)
  - translation-direction error (degrees)
- **ATE (Absolute Trajectory Error, RMSE)** vs ground truth for:
  - **N = 2** keyframes
  - **N = 4** keyframes
  - both **SE(3)** and **Sim(3)** alignment

## Inputs (paths in this repo)

Ground truth:
- `data/middlebury_templeRing/templeRing/templeR_par.txt`

Estimated outputs (samples):
- Python:
  - `samples/python/keyframes_camera_centers.csv`
  - `samples/python/posegraph_edges.csv`
- C++:
  - `samples/cpp/keyframes_camera_centers.csv`
  - `samples/cpp/posegraph_edges.csv`

Benchmark artifacts generated and stored:
- `out/bench/ate_results_raw.txt`
- `out/bench/ate_results_table.md`
- `out/bench/edge_results_raw.txt`
- `out/bench/edge_results_table.md`
- `out/bench/cpp_keyframes_filtered_N2.csv`
- `out/bench/cpp_keyframes_filtered_N4.csv`

## Keyframes used

Keyframe image names are taken from the Python keyframe CSV ordering.

**N = 2**
- `templeR0001.png`
- `templeR0003.png`

**N = 4**
- `templeR0001.png`
- `templeR0003.png`
- `templeR0005.png`
- `templeR0006.png`

## How the benchmarks are computed

Evaluation is performed using the C++ evaluation tools built from `cpp/`:

- ATE: `cpp/build/eval_ate` (SE(3) and Sim(3))
- Keyframe edge error: `cpp/build/gt_keyframe_edge` (edge 0→1)

Full command lines and raw console outputs are captured verbatim in:
- `out/bench/ate_results_raw.txt`
- `out/bench/edge_results_raw.txt`

## Results — ATE (RMSE) vs ground truth

# Stage 2 — ATE results (existing outputs, no rerun)
Ground truth: `data/middlebury_templeRing/templeRing/templeR_par.txt`

Keyframes used (by Python ordering):
- N=2: `templeR0001.png`, `templeR0003.png`
- N=4: `templeR0001.png`, `templeR0003.png`, `templeR0005.png`, `templeR0006.png`

| N keyframes | Alignment | Python ATE_RMSE | C++ ATE_RMSE |
|---:|:---:|---:|---:|
| 2 | Sim(3) | 3.395220e-11 | 3.526060e-11 |
| 2 | SE(3) | 1.852890e-08 | 1.852890e-08 |
| 4 | Sim(3) | 0.014036 | 0.014036 |
| 4 | SE(3) | 0.014772 | 0.014772 |

## Results — Keyframe edge error (0→1) vs ground truth

# Stage 3 — Keyframe edge error (ground truth)

Edge evaluated: **i=0 → j=1** (images: `templeR0001.png` → `templeR0003.png`).

| Metric | Python | C++ |
|---|---:|---:|
| Rotation error (deg) | 5.039282 | 5.039282 |
| Translation direction error (deg) | 3.938996 | 3.938996 |

> Note: In this bundle, the Python and C++ sample `posegraph_edges.csv` entries for edge 0→1 are identical, so the two error values match exactly.

## Notes and caveats

- The sample keyframe lists in `samples/python/` and `samples/cpp/` are identical in this bundle (same `image` ordering).
- The keyframe-edge results match exactly because the sampled **edge 0→1** entries in `samples/python/posegraph_edges.csv` and `samples/cpp/posegraph_edges.csv` are identical in this bundle.
- If you want to benchmark *fresh runs* (rather than the shipped samples), first regenerate:
  - `out/python/keyframes_camera_centers.csv`, `out/python/posegraph_edges.csv`
  - `out/cpp/keyframes_camera_centers.csv`, `out/cpp/posegraph_edges.csv`
  and then rerun the evaluation tools against those files.
