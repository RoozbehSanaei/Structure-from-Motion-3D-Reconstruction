# Stage 1 — Keyframe alignment map
## Source files
- Python keyframes: `samples/python/keyframes_camera_centers.csv`
- C++ keyframes: `samples/cpp/keyframes_camera_centers.csv`

## Image-name column
- `image`

## Keyframe image lists
### Python (order used for benchmarks)
- 0: `templeR0001.png`
- 1: `templeR0003.png`
- 2: `templeR0005.png`
- 3: `templeR0006.png`
- 4: `templeR0008.png`
- 5: `templeR0010.png`
- 6: `templeR0011.png`
- 7: `templeR0012.png`

### C++
- 0: `templeR0001.png`
- 1: `templeR0003.png`
- 2: `templeR0005.png`
- 3: `templeR0006.png`
- 4: `templeR0008.png`
- 5: `templeR0010.png`
- 6: `templeR0011.png`
- 7: `templeR0012.png`

## Intersection
- Count: 8 / Python 8 / C++ 8
- Images:
  - `templeR0001.png`
  - `templeR0003.png`
  - `templeR0005.png`
  - `templeR0006.png`
  - `templeR0008.png`
  - `templeR0010.png`
  - `templeR0011.png`
  - `templeR0012.png`

## Benchmark subsets (by Python ordering)
### N = 2
- `templeR0001.png`
- `templeR0003.png`

### N = 4
- `templeR0001.png`
- `templeR0003.png`
- `templeR0005.png`
- `templeR0006.png`

## Generated filtered C++ files
- `cpp_keyframes_filtered_N2.csv`
- `cpp_keyframes_filtered_N4.csv`
