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
