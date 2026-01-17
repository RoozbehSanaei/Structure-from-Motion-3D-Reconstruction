## 1) What the pipeline takes in, and what it produces

**Input**
- A sequence of images (frames) of the same scene taken from different camera positions.
- Camera calibration (intrinsics) such as focal length and principal point, usually represented by a matrix **K**.

**Output**
- An estimated **camera trajectory**: where the camera was for selected frames (keyframes).
- A **sparse 3D point cloud**: a set of 3D points that correspond to tracked image features.

---

## 2) Core idea: “Track points across images, then infer motion and 3D”

The pipeline repeats this loop:

1. Track 2D feature points from frame to frame.
2. Use the tracked 2D correspondences to estimate the camera’s relative motion between two frames.
3. Convert that motion + correspondences into 3D points (triangulation).
4. Regularly refine everything locally (bundle adjustment).
5. Occasionally correct long-term drift using loop closure + pose graph optimization.

---

## 3) Feature tracking with KLT (stability)

### KLT tracking
**KLT (Kanade–Lucas–Tomasi)** tracking is an **optical flow** method (it estimates how small patches of pixels move between frames). Instead of detecting and matching features from scratch every frame, it tries to **follow the same points** over time.

Why it helps:
- Long-lived tracks reduce jitter and mismatches.
- Motion estimation becomes more stable because correspondences are consistent.

### Forward–backward check
A common filter is **forward–backward consistency**:
- Track a point from frame *t* to *t+1* (forward).
- Track it back from *t+1* to *t* (backward).
- If it doesn’t return close to the original location, it’s likely unreliable and gets dropped.

### Track replenishment
If too many tracks are lost (occlusions, blur, leaving the image), the pipeline **detects new features** and starts new tracks so enough correspondences remain available.

---

## 4) Estimating relative motion using epipolar geometry

Once correspondences between two frames exist (the same track IDs observed in both images), the relative pose can be estimated.

### Normalized coordinates
Using intrinsics **K**, pixels are mapped to **normalized image coordinates** (approximately “ray directions” in the camera frame). This removes the effect of focal length and principal point.

### Essential matrix (E)
The **Essential matrix (E)** is a 3×3 matrix that encodes the geometric constraint between two calibrated views. If a correspondence is correct, it should satisfy the **epipolar constraint**:
- In simple terms: the point in image 2 must lie on a specific line (the **epipolar line**) predicted by the point in image 1 and E.

### RANSAC (robust to outliers)
Real correspondences include some wrong matches (**outliers**).  
**RANSAC (Random Sample Consensus)** handles this by:
1. Randomly sampling a minimal set of correspondences.
2. Fitting a model (E).
3. Counting how many correspondences agree with it (the **inliers**, based on an error threshold).
4. Repeating many times and keeping the model with the most inliers.

The output is:
- Best E
- Set of inlier correspondences

### Decomposition: rotation and translation
From E, relative pose is recovered:
- **R**: rotation (how the camera turned)
- **t**: translation direction (up to an unknown scale)

Important limitation:
- **Scale ambiguity**: with a single camera, translation magnitude cannot be recovered from geometry alone. The trajectory is correct up to an overall scale factor.

---

## 5) Triangulation (creating 3D points)

Given two camera poses and a correspondence, **triangulation** estimates the 3D point that best explains where that feature appears in both images.

Common method:
- **DLT (Direct Linear Transform)** triangulation: solves a linear system derived from the projection equations.

Practical checks:
- **Cheirality**: the point should be in front of both cameras (positive depth).
- Reprojection error should be reasonable (project the 3D point back into the images and see if it lands near the observed pixels).

The pipeline builds a **sparse map**:
- Each **track** that survives long enough can become a 3D **map point**.

---

## 6) Keyframes (keep computation bounded)

Not every frame is kept forever.

A **keyframe** is a selected frame that becomes part of the persistent map/trajectory. Selection is based on signals like:
- Enough **parallax**: parallax is the apparent shift of features between views; bigger parallax generally improves triangulation accuracy.
- Inlier count drops (motion is harder or tracking degraded).
- Time since last keyframe.

Keyframes help because:
- Bundle adjustment and mapping costs stay manageable.
- Geometry stays well-conditioned (informative viewpoints are retained).

---

## 7) Local Bundle Adjustment (accuracy boost)

### Bundle Adjustment (BA)
**Bundle Adjustment** jointly optimizes:
- Camera poses (R, t for keyframes)
- 3D point positions

Objective:
- Minimize **reprojection error** (difference between observed 2D feature locations and the projected location of the optimized 3D point).

### Sliding window BA
Full BA over all frames and points is expensive.  
A **sliding window** approach optimizes only:
- The last N keyframes
- The active subset of points observed by those keyframes

### Robust loss (Huber)
Outliers can remain. A **Huber loss** is a robust cost:
- Quadratic for small errors (like least squares)
- Linear for large errors (down-weights large residuals)

Why BA helps:
- Small errors accumulate over time.
- BA redistributes error consistently across poses and points, improving accuracy.

---

## 8) Loop Closure + Pose Graph (drift reduction)

### Drift
Each step has small estimation error, so the trajectory slowly **drifts** over time (accumulated error).

### Loop closure detection
A **loop closure** occurs when the camera returns to a previously seen place.
Detection often uses:
- Image descriptors (global or local) to find similar keyframes.
- Geometric verification (Essential matrix + RANSAC) to confirm a true match.

### Pose graph
A **pose graph** is a graph where:
- Nodes = keyframe poses
- Edges = relative pose constraints between keyframes  
  - Sequential edges: between consecutive keyframes  
  - Loop edges: between distant keyframes that match the same place

### Pose graph optimization
Given many relative constraints, a global optimization finds poses that best satisfy all edges (least-squares over pose parameters).
Effect:
- Loop edges “pull” the trajectory back into alignment.
- Drift is reduced without re-optimizing every 3D point (cheaper than full global BA).

---

## 9) What “accuracy” means here

Higher accuracy typically means:
- Lower **reprojection error**
- More consistent trajectory (less drift)
- Cleaner point cloud (fewer outliers, better structure)
- Better loop closure consistency

These improvements contribute directly:
- **KLT tracks** → better correspondences → better E estimation
- **Local BA** → reduces pose/point error in recent history
- **Loop closure + pose graph** → reduces global drift

---

## 10) Known limitations (normal for monocular SfM)

- **Unknown scale** (monocular ambiguity): results are up to a scale factor.
- **Low parallax segments** (camera mostly rotating or moving slowly) are hard: triangulation becomes unstable.
- **Textureless or repetitive patterns** cause tracking/matching failures.
- Global optimality is not guaranteed: RANSAC and local optimizations can settle into local minima.
