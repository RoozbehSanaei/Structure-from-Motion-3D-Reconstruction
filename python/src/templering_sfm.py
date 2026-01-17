#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
templering_klt_ba_loop_slam_v3.py

Single-file, end-to-end classic monocular reconstruction with:
  • KLT feature tracks (stable correspondences)
  • Sliding-window local bundle adjustment (robust LM)
  • Loop closure + pose-graph optimization (drift reduction)

This version emphasizes modern Python:
  • dataclasses (slots/frozen), StrEnum, cached_property
  • rich typing (TypeAlias, NDArray, Self, Literal, overload-friendly APIs)
  • match/case control flow, pathlib throughout
  • context managers + structured logging
  • minimal global state; cohesive classes + clear interfaces

Works best on Middlebury TempleRing (includes per-image calibration in templeR_par.txt).
For generic image sequences/videos, provide K via OpenCV YAML with node "K".

Dependencies:
  pip install opencv-python numpy pandas matplotlib pillow

Examples:
  # TempleRing from zip
  curl -L -o templeRing.zip https://vision.middlebury.edu/mview/data/data/templeRing.zip
  python templering_klt_ba_loop_slam_v3.py --zip templeRing.zip --frames 12 --use-gt-scale --translation-mode full --visuals

  # TempleRing already extracted (root contains ./templeRing/templeR_par.txt)
  python templering_klt_ba_loop_slam_v3.py --dir templeRing --frames 12 --use-gt-scale --translation-mode full --visuals

Outputs (in --out):
  - templeRing_sparse_points.ply
  - keyframes_camera_centers.csv
  - posegraph_edges.csv
  - (optional) input_montage.png, inlier_matches.png, sparse_pointcloud.png, camera_trajectory.png
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import json
import zipfile
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final, Iterable, TypeAlias, Self

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image, ImageDraw


# -----------------------------
# Typing aliases
# -----------------------------
F64: TypeAlias = np.float64
U8: TypeAlias = np.uint8
F32: TypeAlias = np.float32

Mat33: TypeAlias = NDArray[F64]
Vec3: TypeAlias = NDArray[F64]
Pts2: TypeAlias = NDArray[F32]  # (N,2)


# -----------------------------
# Small numeric helpers
# -----------------------------
def as_f64(x) -> NDArray[F64]:
    return np.asarray(x, dtype=np.float64)


def unit(v: Vec3, eps: float = 1e-12) -> Vec3:
    n = float(np.linalg.norm(v))
    return (v * 0.0) if n < eps else (v / n)


def rot_log(R: Mat33) -> Vec3:
    rvec, _ = cv2.Rodrigues(R.astype(np.float64, copy=False))
    return rvec.reshape(3).astype(np.float64, copy=False)


def rot_exp(rvec: Vec3) -> Mat33:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R.astype(np.float64, copy=False)


# -----------------------------
# Pose: Camera-to-World (C->W)
# -----------------------------
@dataclass(frozen=True, slots=True)
class PoseCW:
    """
    Camera-to-world transform: Xw = R * Xc + t
    """
    R: Mat33
    t: Vec3  # (3,)

    @staticmethod
    def I() -> Self:
        return PoseCW(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    def inv(self) -> "PoseWC":
        # world->camera: Xc = R^T Xw - R^T t
        Rwc = self.R.T
        twc = -(Rwc @ self.t)
        return PoseWC(Rwc, twc)

    def compose_right_inv_ij(self, R_ji: Mat33, t_ji: Vec3) -> Self:
        """
        We estimate relative pose i->j: Xj = R_ji Xi + t_ji (in cam-i coordinates).
        To update current camera-to-world pose, we apply the inverse (j->i) on the right:
          Xi = R_ji^T Xj - R_ji^T t_ji
        """
        R_delta = R_ji.T
        t_delta = -(R_ji.T @ t_ji)
        R_new = self.R @ R_delta
        t_new = (self.R @ t_delta) + self.t
        return PoseCW(R_new, t_new)

    def rvec_t(self) -> tuple[Vec3, Vec3]:
        rvec, _ = cv2.Rodrigues(self.R)
        return rvec.reshape(3).astype(np.float64, copy=False), self.t.reshape(3).astype(np.float64, copy=False)

    @staticmethod
    def from_rvec_t(rvec: Vec3, t: Vec3) -> Self:
        return PoseCW(rot_exp(rvec), np.asarray(t, dtype=np.float64).reshape(3))


@dataclass(frozen=True, slots=True)
class PoseWC:
    """
    World-to-camera transform: Xc = R * Xw + t
    """
    R: Mat33
    t: Vec3

    def P(self, K: Mat33) -> NDArray[F64]:
        return K @ np.hstack([self.R, self.t.reshape(3, 1)])


# -----------------------------
# TempleRing dataset loader
# -----------------------------
@dataclass(frozen=True, slots=True)
class MiddleburyRecord:
    img: str
    K: Mat33
    R: Mat33
    t: Vec3  # (3,)

    @property
    def pose_wc(self) -> PoseWC:
        return PoseWC(self.R, self.t)

    @property
    def pose_cw(self) -> PoseCW:
        return PoseCW(self.R.T, -(self.R.T @ self.t))


@dataclass(frozen=True, slots=True)
class MiddleburyAngles:
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class TempleRing:
    root: Path
    records: tuple[MiddleburyRecord, ...]
    angles: dict[str, MiddleburyAngles]

    @property
    def data_dir(self) -> Path:
        return self.root / "templeRing"

    @property
    def par_file(self) -> Path:
        return self.data_dir / "templeR_par.txt"

    @property
    def ang_file(self) -> Path:
        return self.data_dir / "templeR_ang.txt"

    def image_path(self, name: str) -> Path:
        return self.data_dir / name

    @staticmethod
    def _read_par(path: Path) -> tuple[MiddleburyRecord, ...]:
        with path.open("r", encoding="utf-8") as f:
            n = int(f.readline().strip())
            out: list[MiddleburyRecord] = []
            for _ in range(n):
                toks = f.readline().split()
                img = toks[0]
                nums = np.array([float(x) for x in toks[1:]], dtype=np.float64)
                K = nums[0:9].reshape(3, 3)
                R = nums[9:18].reshape(3, 3)
                t = nums[18:21].reshape(3)
                out.append(MiddleburyRecord(img=img, K=K, R=R, t=t))
        return tuple(out)

    @staticmethod
    def _read_ang(path: Path) -> dict[str, MiddleburyAngles]:
        out: dict[str, MiddleburyAngles] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                lat_s, lon_s, img = line.split()
                out[img] = MiddleburyAngles(lat=float(lat_s), lon=float(lon_s))
        return out

    @classmethod
    def from_zip(cls, zip_path: Path, extract_to: Path) -> Self:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)
        tmp = cls(root=extract_to, records=tuple(), angles={})
        return cls(root=extract_to, records=cls._read_par(tmp.par_file), angles=cls._read_ang(tmp.ang_file))

    @classmethod
    def from_dir(cls, root: Path) -> Self:
        tmp = cls(root=root, records=tuple(), angles={})
        return cls(root=root, records=cls._read_par(tmp.par_file), angles=cls._read_ang(tmp.ang_file))


def load_K_yaml(yaml_path: Path) -> Mat33:
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    try:
        if not fs.isOpened():
            raise FileNotFoundError(f"Could not open YAML: {yaml_path}")
        K = fs.getNode("K").mat()
        if K is None:
            raise ValueError("YAML must contain node 'K' as OpenCV matrix.")
        return np.asarray(K, dtype=np.float64)
    finally:
        fs.release()


# -----------------------------
# Config enums and dataclasses
# -----------------------------
class TranslationMode(StrEnum):
    FULL = "full"   # residual on translation vector
    DIR = "dir"     # residual on translation direction
    ROT = "rot"     # rotation-only constraints



class ExportGeometry(StrEnum):
    NONE = "none"               # do not export geometry
    POINTCLOUD = "pointcloud"   # export sparse triangulated points
    MESH_STEREO = "mesh_stereo"  # export a mesh from a stereo depth map (keyframe pair)
    BOTH = "both"               # export both pointcloud and mesh


@dataclass(frozen=True, slots=True)
class StereoMeshConfig:
    # Subsample factor for the rectified disparity grid (higher = smaller mesh)
    step: int = 4
    # StereoSGBM parameters (keep minimal but stable)
    num_disparities: int = 128  # must be divisible by 16
    block_size: int = 7         # odd number, typical 5..11
    disp_min: float = 1.0       # reject disparities below this value
    disp_jump: float = 3.0      # reject triangles spanning large disparity jumps
    z_max_percentile: float = 98.0  # drop extreme depth outliers using percentile in rectified frame

@dataclass(frozen=True, slots=True)
class KLTConfig:
    max_tracks: int = 2200
    min_tracks: int = 900
    quality: float = 0.01
    min_distance: int = 8
    block_size: int = 7
    win_size: tuple[int, int] = (21, 21)
    max_level: int = 3
    fb_thresh: float = 1.0


@dataclass(frozen=True, slots=True)
class BAConfig:
    window: int = 5
    max_points: int = 200
    iters: int = 6
    lambda0: float = 1e-2
    huber_delta: float = 3.0
    eps_pose: float = 1e-6


@dataclass(frozen=True, slots=True)
class LoopConfig:
    min_kf_gap: int = 8
    top_k: int = 5
    min_matches: int = 80
    min_inliers: int = 60


@dataclass(frozen=True, slots=True)
class PoseGraphConfig:
    iters: int = 10
    lambda0: float = 1e-2
    eps: float = 1e-6
    mode: TranslationMode = TranslationMode.DIR
    w_rot: float = 1.0
    w_trans: float = 1.0


@dataclass(frozen=True, slots=True)
class SystemConfig:
    frames: int = 12
    use_gt_scale: bool = False
    translation_mode: TranslationMode = TranslationMode.DIR

    # keyframe policy
    min_inliers: int = 200
    keyframe_min_gap: int = 1
    keyframe_parallax_px: float = 18.0

    # optimization cadence
    loop_every_kf: int = 1
    posegraph_every_kf: int = 1

    # sub-configs
    klt: KLTConfig = KLTConfig()
    ba: BAConfig = BAConfig()
    loop: LoopConfig = LoopConfig()
    pg: PoseGraphConfig = PoseGraphConfig()


# -----------------------------
# Shared config.json loading
# -----------------------------

def _deep_merge(a: dict, b: dict) -> dict:
    """Recursively merge dicts. Values in b win."""
    out = dict(a)
    for k, vb in b.items():
        va = out.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out[k] = _deep_merge(va, vb)
        else:
            out[k] = vb
    return out


def _repo_root() -> Path:
    # .../python/src/templering_sfm.py -> repo root
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return _repo_root() / 'config.json'


def _load_config_json(path: Path | None, log: logging.Logger) -> dict:
    if path is None:
        return {}
    if not path.exists():
        log.warning('Config file not found: %s (using built-in defaults)', path)
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('Top-level JSON must be an object')
        return data
    except Exception as e:
        raise SystemExit(f'Failed to load config.json ({path}): {e}')


def _cfg_get(d: dict, keys: list[str], default):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _first_not_none(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

# -----------------------------
# KLT tracker with stable IDs
# -----------------------------
class KLTTracker:
    def __init__(self, cfg: KLTConfig):
        self.cfg = cfg
        self._next_id: int = 0
        self._prev_gray: NDArray[U8] | None = None
        self.pts: Pts2 = np.empty((0, 2), np.float32)
        self.ids: NDArray[np.int64] = np.empty((0,), np.int64)

    def _detect(self, gray: NDArray[U8], needed: int) -> Pts2:
        if needed <= 0:
            return np.empty((0, 2), np.float32)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for x, y in self.pts.astype(np.int32):
            cv2.circle(mask, (int(x), int(y)), self.cfg.min_distance, 0, -1)
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(needed),
            qualityLevel=float(self.cfg.quality),
            minDistance=float(self.cfg.min_distance),
            blockSize=int(self.cfg.block_size),
            mask=mask,
        )
        return (pts.reshape(-1, 2).astype(np.float32) if pts is not None else np.empty((0, 2), np.float32))

    def reset(self, gray: NDArray[U8]) -> None:
        self._prev_gray = gray
        self.pts = self._detect(gray, self.cfg.max_tracks)
        n = self.pts.shape[0]
        self.ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n

    def step(self, gray: NDArray[U8]) -> tuple[Pts2, Pts2, NDArray[np.int64]]:
        if self._prev_gray is None:
            self.reset(gray)
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.int64)

        if self.pts.size == 0:
            self.reset(gray)
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32), np.empty((0,), np.int64)

        p0 = self.pts.reshape(-1, 1, 2)
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, p0, None,
            winSize=self.cfg.win_size, maxLevel=self.cfg.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._prev_gray, p1, None,
            winSize=self.cfg.win_size, maxLevel=self.cfg.max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        st1 = st1.reshape(-1).astype(bool)
        st2 = st2.reshape(-1).astype(bool)
        fb = np.linalg.norm(p0r.reshape(-1, 2) - self.pts, axis=1)
        ok = st1 & st2 & (fb < self.cfg.fb_thresh)

        prev_pts = self.pts[ok]
        cur_pts = p1.reshape(-1, 2)[ok].astype(np.float32, copy=False)
        ids = self.ids[ok].copy()

        # update state
        self._prev_gray = gray
        self.pts = cur_pts
        self.ids = ids

        # replenish
        if self.pts.shape[0] < self.cfg.min_tracks:
            add = self._detect(gray, self.cfg.max_tracks - self.pts.shape[0])
            if add.shape[0] > 0:
                new_ids = np.arange(self._next_id, self._next_id + add.shape[0], dtype=np.int64)
                self._next_id += add.shape[0]
                self.pts = np.vstack([self.pts, add])
                self.ids = np.hstack([self.ids, new_ids])

        return prev_pts, cur_pts, ids


# -----------------------------
# Map and keyframes
# -----------------------------
@dataclass(frozen=True, slots=True)
class Keyframe:
    kf_id: int
    frame_idx: int
    img_name: str
    img_bgr: NDArray[U8]
    gray: NDArray[U8]
    pose_cw: PoseCW
    obs: dict[int, NDArray[F32]]  # track_id -> uv
    orb_kps: tuple[cv2.KeyPoint, ...]
    orb_des: NDArray[U8]


@dataclass(slots=True)
class MapPoint:
    pid: int
    track_id: int
    Xw: Vec3
    obs: list[tuple[int, NDArray[F32]]] = field(default_factory=list)


class MapState:
    def __init__(self) -> None:
        self._next_pid = 0
        self.track_to_pid: dict[int, int] = {}
        self.points: dict[int, MapPoint] = {}

    def has_track(self, tid: int) -> bool:
        return tid in self.track_to_pid

    def pid_for(self, tid: int) -> int | None:
        return self.track_to_pid.get(tid)

    def add_point(self, tid: int, Xw: Vec3) -> int:
        pid = self._next_pid
        self._next_pid += 1
        mp = MapPoint(pid=pid, track_id=tid, Xw=Xw.reshape(3).astype(np.float64, copy=False))
        self.points[pid] = mp
        self.track_to_pid[tid] = pid
        return pid

    def add_obs(self, tid: int, kf_id: int, uv: NDArray[F32]) -> None:
        pid = self.pid_for(tid)
        if pid is None:
            return
        self.points[pid].obs.append((kf_id, uv.astype(np.float32, copy=False)))

    def xyz(self) -> NDArray[F64]:
        if not self.points:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack([mp.Xw for mp in self.points.values()]).astype(np.float64, copy=False)


# -----------------------------
# Loop closure: ORB + geometric verify
# -----------------------------
class LoopClosure:
    def __init__(self, cfg: LoopConfig):
        self.cfg = cfg
        self.orb = cv2.ORB_create(nfeatures=4000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def compute(self, gray: NDArray[U8]) -> tuple[tuple[cv2.KeyPoint, ...], NDArray[U8]]:
        kps, des = self.orb.detectAndCompute(gray, None)
        if des is None:
            return tuple(), np.empty((0, 32), np.uint8)
        return tuple(kps), des

    def _ratio(self, des1: NDArray[U8], des2: NDArray[U8], ratio: float = 0.75) -> list[cv2.DMatch]:
        if des1.size == 0 or des2.size == 0:
            return []
        knn = self.bf.knnMatch(des1, des2, k=2)
        good: list[cv2.DMatch] = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    def propose_edges(self, K: Mat33, keyframes: list[Keyframe], new_kf: Keyframe) -> list[tuple[int, int, Mat33, Vec3]]:
        if not keyframes:
            return []
        j = new_kf.kf_id
        cands = [kf for kf in keyframes if (j - kf.kf_id) >= self.cfg.min_kf_gap]
        if not cands:
            return []

        scored = []
        for kf in cands:
            scored.append((len(self._ratio(kf.orb_des, new_kf.orb_des)), kf))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [kf for s, kf in scored[: self.cfg.top_k] if s >= self.cfg.min_matches]
        if not top:
            return []

        edges: list[tuple[int, int, Mat33, Vec3]] = []
        for kf in top:
            ms = self._ratio(kf.orb_des, new_kf.orb_des)
            if len(ms) < self.cfg.min_matches:
                continue
            pts_i = np.float32([kf.orb_kps[m.queryIdx].pt for m in ms])
            pts_j = np.float32([new_kf.orb_kps[m.trainIdx].pt for m in ms])

            E, inl = cv2.findEssentialMat(pts_i, pts_j, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None or inl is None:
                continue
            in_mask = inl.ravel().astype(bool)
            if int(in_mask.sum()) < self.cfg.min_inliers:
                continue

            pts_i_in = pts_i[in_mask]
            pts_j_in = pts_j[in_mask]
            in_pose, R_ji, t_ji, _ = cv2.recoverPose(E, pts_i_in, pts_j_in, K)
            if int(in_pose) < self.cfg.min_inliers:
                continue

            edges.append((kf.kf_id, new_kf.kf_id, R_ji.astype(np.float64, copy=False), t_ji.reshape(3).astype(np.float64, copy=False)))
        return edges


# -----------------------------
# Pose graph optimization (small graphs; numeric Jacobian)
# -----------------------------
@dataclass(frozen=True, slots=True)
class Edge:
    i: int
    j: int
    R_ji: Mat33
    t_ji: Vec3
    w_rot: float = 1.0
    w_trans: float = 1.0


class PoseGraph:
    def __init__(self, cfg: PoseGraphConfig):
        self.cfg = cfg

    @staticmethod
    def rel_pred(p_i: PoseCW, p_j: PoseCW) -> tuple[Mat33, Vec3]:
        # predicted i->j in cam-i coords:
        # R_ji = R_wc_j * R_cw_i = R_j^T * R_i
        R_ji = p_j.R.T @ p_i.R
        t_ji = p_j.R.T @ (p_i.t - p_j.t)
        return R_ji, t_ji

    def edge_residual(self, p_i: PoseCW, p_j: PoseCW, e: Edge) -> NDArray[F64]:
        R_pred, t_pred = self.rel_pred(p_i, p_j)
        rR = rot_log(e.R_ji.T @ R_pred) * e.w_rot

        mode = self.cfg.mode
        match mode:
            case TranslationMode.ROT:
                return rR
            case TranslationMode.DIR:
                rt = (unit(t_pred) - unit(e.t_ji)) * e.w_trans
                return np.hstack([rR, rt])
            case TranslationMode.FULL:
                rt = (t_pred - e.t_ji) * e.w_trans
                return np.hstack([rR, rt])
            case _:
                return rR

    def optimize(self, poses: list[PoseCW], edges: list[Edge]) -> list[PoseCW]:
        if len(poses) <= 2 or not edges:
            return poses

        N = len(poses)
        x0 = []
        for i in range(1, N):
            rv, t = poses[i].rvec_t()
            x0.append(rv); x0.append(t)
        x = np.hstack(x0).astype(np.float64, copy=False)

        def unpack(xv: NDArray[F64]) -> list[PoseCW]:
            out = [poses[0]]
            off = 0
            for _ in range(1, N):
                rv = xv[off:off+3]; tv = xv[off+3:off+6]; off += 6
                out.append(PoseCW.from_rvec_t(rv, tv))
            return out

        def residual(xv: NDArray[F64]) -> NDArray[F64]:
            ps = unpack(xv)
            rs = [self.edge_residual(ps[e.i], ps[e.j], e) for e in edges]
            return np.concatenate(rs).astype(np.float64, copy=False)

        lam = float(self.cfg.lambda0)
        r0 = residual(x)
        best = float(0.5 * r0 @ r0)
        eps = float(self.cfg.eps)

        for _ in range(self.cfg.iters):
            r0 = residual(x)
            M = r0.size
            D = x.size
            J = np.zeros((M, D), dtype=np.float64)
            for k in range(D):
                xk = x.copy()
                xk[k] += eps
                rk = residual(xk)
                J[:, k] = (rk - r0) / eps

            A = J.T @ J + lam * np.eye(D, dtype=np.float64)
            b = -J.T @ r0
            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(A, b, rcond=None)[0]

            x_try = x + dx
            r_try = residual(x_try)
            cost_try = float(0.5 * (r_try @ r_try))
            if cost_try < best:
                x = x_try
                best = cost_try
                lam *= 0.3
            else:
                lam *= 2.0

            if float(np.linalg.norm(dx)) < 1e-6:
                break

        return unpack(x)


# -----------------------------
# Local BA (LM, robust Huber). Numeric pose Jacobian + analytic point Jacobian.
# -----------------------------
class LocalBA:
    def __init__(self, cfg: BAConfig):
        self.cfg = cfg

    @staticmethod
    def project(K: Mat33, pose_cw: PoseCW, Xw: Vec3) -> tuple[NDArray[F64], float, Vec3, Mat33]:
        pose_wc = pose_cw.inv()
        Xc = (pose_wc.R @ Xw) + pose_wc.t
        z = float(Xc[2])
        if z <= 1e-9:
            return np.array([np.nan, np.nan], dtype=np.float64), z, Xc, pose_wc.R
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        u = fx * float(Xc[0]) / z + cx
        v = fy * float(Xc[1]) / z + cy
        return np.array([u, v], dtype=np.float64), z, Xc, pose_wc.R

    def optimize(self, K: Mat33, keyframes: list[Keyframe], m: MapState, active_ids: list[int]) -> list[PoseCW] | None:
        if len(active_ids) < 3:
            return None

        kf_by_id = {kf.kf_id: kf for kf in keyframes}
        window = [kf_by_id[i] for i in active_ids]
        idx_of = {kf.kf_id: wi for wi, kf in enumerate(window)}

        # select points observed >=2 times in window
        cand_pids = []
        for pid, mp in m.points.items():
            cnt = sum(1 for (kf_id, _) in mp.obs if kf_id in idx_of)
            if cnt >= 2:
                cand_pids.append(pid)
        if not cand_pids:
            return None
        cand_pids = cand_pids[: self.cfg.max_points]

        fixed_pose = window[0].pose_cw
        var_kfs = window[1:]
        P = len(cand_pids)
        F = len(var_kfs)

        x_pose0 = []
        for kf in var_kfs:
            rv, t = kf.pose_cw.rvec_t()
            x_pose0.append(rv); x_pose0.append(t)
        x_pose0 = np.hstack(x_pose0) if x_pose0 else np.zeros((0,), np.float64)

        X0 = np.vstack([m.points[pid].Xw for pid in cand_pids]).astype(np.float64, copy=False).reshape(-1)
        x = np.hstack([x_pose0, X0]).astype(np.float64, copy=False)

        obs = []
        for p_i, pid in enumerate(cand_pids):
            for kf_id, uv in m.points[pid].obs:
                if kf_id in idx_of:
                    obs.append((idx_of[kf_id], p_i, uv.astype(np.float32, copy=False)))

        if len(obs) < 20:
            return None

        def unpack(xv: NDArray[F64]) -> tuple[list[PoseCW], NDArray[F64]]:
            poses = [fixed_pose]
            off = 0
            for _ in range(F):
                rv = xv[off:off+3]; tv = xv[off+3:off+6]; off += 6
                poses.append(PoseCW.from_rvec_t(rv, tv))
            X = xv[off:].reshape(P, 3).astype(np.float64, copy=False)
            return poses, X

        def build(xv: NDArray[F64]) -> tuple[NDArray[F64], NDArray[F64]]:
            poses, X = unpack(xv)
            M = len(obs)
            r = np.zeros((2*M,), dtype=np.float64)
            J = np.zeros((2*M, xv.size), dtype=np.float64)

            delta = float(self.cfg.huber_delta)
            eps = float(self.cfg.eps_pose)

            for k, (f_idx, p_i, uv) in enumerate(obs):
                pose = poses[f_idx]
                uv_hat, z, Xc, R_wc = self.project(K, pose, X[p_i])
                if not np.isfinite(uv_hat).all():
                    continue
                e = (uv_hat - uv.astype(np.float64))
                en = float(np.linalg.norm(e))
                w = 1.0 if en <= delta else (delta / (en + 1e-12))
                sw = math.sqrt(w)

                r[2*k:2*k+2] = sw * e

                x_, y_, z_ = float(Xc[0]), float(Xc[1]), float(Xc[2])
                fx, fy = float(K[0, 0]), float(K[1, 1])
                Jproj = np.array(
                    [[fx/z_, 0.0, -fx*x_/(z_*z_)],
                     [0.0, fy/z_, -fy*y_/(z_*z_)]],
                    dtype=np.float64
                )
                # analytic w.r.t point
                p_off = 6*F + 3*p_i
                J[2*k:2*k+2, p_off:p_off+3] = sw * (Jproj @ R_wc)

                # numeric w.r.t pose (if variable)
                if f_idx >= 1:
                    off_pose = 6*(f_idx-1)
                    rv0, t0 = pose.rvec_t()
                    base = np.hstack([rv0, t0])
                    for j in range(6):
                        b2 = base.copy()
                        b2[j] += eps
                        pose2 = PoseCW.from_rvec_t(b2[:3], b2[3:])
                        uv2, z2, *_ = self.project(K, pose2, X[p_i])
                        if not np.isfinite(uv2).all() or z2 <= 1e-9:
                            continue
                        de = (uv2 - uv_hat) / eps
                        J[2*k:2*k+2, off_pose + j] = sw * de

            return r, J

        lam = float(self.cfg.lambda0)
        r0, J0 = build(x)
        best = float(0.5 * (r0 @ r0))

        for _ in range(self.cfg.iters):
            r0, J0 = build(x)
            A = J0.T @ J0 + lam * np.eye(x.size, dtype=np.float64)
            b = -J0.T @ r0
            try:
                dx = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(A, b, rcond=None)[0]

            x_try = x + dx
            r_try, _ = build(x_try)
            cost_try = float(0.5 * (r_try @ r_try))
            if cost_try < best:
                x = x_try
                best = cost_try
                lam *= 0.3
            else:
                lam *= 2.0

            if float(np.linalg.norm(dx)) < 1e-6:
                break

        poses, X = unpack(x)
        # write points back
        for p_i, pid in enumerate(cand_pids):
            m.points[pid].Xw = X[p_i].reshape(3).astype(np.float64, copy=False)
        return poses


# -----------------------------
# Main system orchestration
# -----------------------------
class ClassicSystem:
    def __init__(self, K: Mat33, cfg: SystemConfig, gt_records: dict[str, MiddleburyRecord] | None):
        self.K = K
        self.cfg = cfg
        self.gt = gt_records

        self.tracker = KLTTracker(cfg.klt)
        self.loop = LoopClosure(cfg.loop)
        self.pg = PoseGraph(PoseGraphConfig(
            iters=cfg.pg.iters, lambda0=cfg.pg.lambda0, eps=cfg.pg.eps,
            mode=cfg.translation_mode, w_rot=cfg.pg.w_rot, w_trans=cfg.pg.w_trans
        ))
        self.ba = LocalBA(cfg.ba)

        self.pose_cw: PoseCW = PoseCW.I()
        self.keyframes: list[Keyframe] = []
        self.map = MapState()
        self.edges: list[Edge] = []

        self.track_hist: dict[int, list[tuple[int, NDArray[F32]]]] = defaultdict(list)
        self.last_kf_frame_idx = -10**9

        self._inlier_debug: tuple[NDArray[U8], NDArray[U8], Pts2, Pts2] | None = None

    @staticmethod
    def median_parallax(a: Pts2, b: Pts2) -> float:
        if a.size == 0:
            return 0.0
        return float(np.median(np.linalg.norm((b - a), axis=1)))

    def scale_translation(self, img_i: str, img_j: str, t: Vec3) -> Vec3:
        if not self.cfg.use_gt_scale or self.gt is None:
            return t
        ri = self.gt.get(img_i)
        rj = self.gt.get(img_j)
        if ri is None or rj is None:
            return t
        ci = ri.pose_cw.t
        cj = rj.pose_cw.t
        baseline = float(np.linalg.norm(cj - ci))
        return unit(t) * baseline

    def estimate_rel(self, pts_i: Pts2, pts_j: Pts2) -> tuple[Mat33 | None, Vec3 | None, int, NDArray[np.bool_]]:
        if pts_i.shape[0] < 8:
            return None, None, 0, np.zeros((pts_i.shape[0],), dtype=bool)
        E, inl = cv2.findEssentialMat(pts_i, pts_j, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or inl is None:
            return None, None, 0, np.zeros((pts_i.shape[0],), dtype=bool)
        in_mask = inl.ravel().astype(bool)
        pi = pts_i[in_mask]
        pj = pts_j[in_mask]
        in_pose, R_ji, t_ji, pm = cv2.recoverPose(E, pi, pj, self.K)
        pm = pm.ravel().astype(bool)
        full = np.zeros_like(in_mask)
        full[np.where(in_mask)[0]] = pm
        return R_ji.astype(np.float64, copy=False), t_ji.reshape(3).astype(np.float64, copy=False), int(in_pose), full

    def should_keyframe(self, inliers: int, parallax: float, frame_idx: int) -> bool:
        if (frame_idx - self.last_kf_frame_idx) < self.cfg.keyframe_min_gap:
            return False
        if inliers < self.cfg.min_inliers:
            return True
        return parallax >= self.cfg.keyframe_parallax_px

    def triangulate_pair(self, kf_i: Keyframe, kf_j: Keyframe, uv_i: NDArray[F32], uv_j: NDArray[F32]) -> Vec3 | None:
        P1 = kf_i.pose_cw.inv().P(self.K)
        P2 = kf_j.pose_cw.inv().P(self.K)
        X_h = cv2.triangulatePoints(P1, P2, uv_i.reshape(2, 1), uv_j.reshape(2, 1))
        Xw = (X_h[:3] / (X_h[3] + 1e-12)).reshape(3).astype(np.float64, copy=False)

        def in_front(kf: Keyframe, X: Vec3) -> bool:
            wc = kf.pose_cw.inv()
            Xc = wc.R @ X + wc.t
            return float(Xc[2]) > 0.0

        return Xw if (in_front(kf_i, Xw) and in_front(kf_j, Xw)) else None

    def triangulate_new_points(self) -> None:
        if len(self.keyframes) < 2:
            return
        for tid, hist in list(self.track_hist.items()):
            if self.map.has_track(tid) or len(hist) < 2:
                continue
            (kf0, uv0), (kfl, uvl) = hist[0], hist[-1]
            if kf0 == kfl:
                continue
            Xw = self.triangulate_pair(self.keyframes[kf0], self.keyframes[kfl], uv0, uvl)
            if Xw is None:
                continue
            self.map.add_point(tid, Xw)
            for kf_id, uv in hist:
                self.map.add_obs(tid, kf_id, uv)

    def add_keyframe(self, frame_idx: int, img_name: str, img_bgr: NDArray[U8], gray: NDArray[U8]) -> None:
        kf_id = len(self.keyframes)
        obs = {int(tid): uv.copy() for tid, uv in zip(self.tracker.ids.tolist(), self.tracker.pts)}
        orb_kps, orb_des = self.loop.compute(gray)

        kf = Keyframe(
            kf_id=kf_id,
            frame_idx=frame_idx,
            img_name=img_name,
            img_bgr=img_bgr,
            gray=gray,
            pose_cw=self.pose_cw,
            obs=obs,
            orb_kps=orb_kps,
            orb_des=orb_des,
        )
        self.keyframes.append(kf)
        self.last_kf_frame_idx = frame_idx

        for tid, uv in obs.items():
            self.track_hist[tid].append((kf_id, uv))
            if self.map.has_track(tid):
                self.map.add_obs(tid, kf_id, uv)

        self.triangulate_new_points()

        # loop closure proposals
        if (kf_id % self.cfg.loop_every_kf) == 0 and kf_id >= 2:
            for i, j, R_ji, t_ji in self.loop.propose_edges(self.K, self.keyframes[:-1], kf):
                self.edges.append(Edge(i=i, j=j, R_ji=R_ji, t_ji=(t_ji if self.cfg.translation_mode == TranslationMode.FULL else unit(t_ji)),
                                       w_rot=self.cfg.pg.w_rot, w_trans=self.cfg.pg.w_trans))

        # pose graph optimize
        if (kf_id % self.cfg.posegraph_every_kf) == 0 and len(self.keyframes) >= 3 and self.edges:
            self.optimize_pose_graph()

        # local BA
        self.local_ba()

    def optimize_pose_graph(self) -> None:
        poses = [kf.pose_cw for kf in self.keyframes]
        poses_opt = self.pg.optimize(poses, self.edges)
        self.keyframes = [
            Keyframe(
                kf_id=kf.kf_id, frame_idx=kf.frame_idx, img_name=kf.img_name,
                img_bgr=kf.img_bgr, gray=kf.gray, pose_cw=p,
                obs=kf.obs, orb_kps=kf.orb_kps, orb_des=kf.orb_des
            )
            for kf, p in zip(self.keyframes, poses_opt)
        ]
        self.pose_cw = self.keyframes[-1].pose_cw

    def local_ba(self) -> None:
        if len(self.keyframes) < 3:
            return
        W = self.cfg.ba.window
        active = [kf.kf_id for kf in self.keyframes[-W:]]
        poses_opt = self.ba.optimize(self.K, self.keyframes, self.map, active)
        if poses_opt is None:
            return
        by_id = {kf.kf_id: kf for kf in self.keyframes}
        for kf_id, p in zip(active, poses_opt):
            kf = by_id[kf_id]
            by_id[kf_id] = Keyframe(
                kf_id=kf.kf_id, frame_idx=kf.frame_idx, img_name=kf.img_name,
                img_bgr=kf.img_bgr, gray=kf.gray, pose_cw=p,
                obs=kf.obs, orb_kps=kf.orb_kps, orb_des=kf.orb_des
            )
        self.keyframes = [by_id[i] for i in range(len(self.keyframes))]
        self.pose_cw = self.keyframes[-1].pose_cw

    def process(self, frame_idx: int, img_name: str, img_bgr: NDArray[U8]) -> None:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        pts_prev, pts_cur, _ids = self.tracker.step(gray)

        if pts_prev.size == 0:
            self.add_keyframe(frame_idx, img_name, img_bgr, gray)
            return

        R_ji, t_ji, inliers, mask = self.estimate_rel(pts_prev, pts_cur)
        if R_ji is None or t_ji is None or inliers < 20:
            self.add_keyframe(frame_idx, img_name, img_bgr, gray)
            return

        p_i = pts_prev[mask]
        p_j = pts_cur[mask]
        parallax = self.median_parallax(p_i, p_j)

        # scale odometry if requested
        last_img = (self.keyframes[-1].img_name if self.keyframes else img_name)
        t_scaled = self.scale_translation(last_img, img_name, t_ji)

        # pose update
        self.pose_cw = self.pose_cw.compose_right_inv_ij(R_ji, t_scaled)

        # keyframe decision
        if self.should_keyframe(inliers, parallax, frame_idx) or not self.keyframes:
            # add odom edge between last and new keyframe
            if self.keyframes:
                i = self.keyframes[-1].kf_id
                j = i + 1
                t_edge = t_scaled if self.cfg.translation_mode == TranslationMode.FULL else unit(t_scaled)
                self.edges.append(Edge(i=i, j=j, R_ji=R_ji, t_ji=t_edge, w_rot=self.cfg.pg.w_rot, w_trans=self.cfg.pg.w_trans))

                # stash a visualization payload once
                if self._inlier_debug is None and p_i.shape[0] >= 50:
                    self._inlier_debug = (self.keyframes[-1].img_bgr, img_bgr, p_i, p_j)

            self.add_keyframe(frame_idx, img_name, img_bgr, gray)

    @property
    def inlier_debug(self):
        return self._inlier_debug


# -----------------------------
# Exports + visuals
# -----------------------------
def write_ply_xyz(path: Path, xyz: NDArray[F64]) -> None:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")




def write_ply_mesh(path: Path, vertices: NDArray[F64], faces: NDArray[np.int64]) -> None:
    """Write a triangle mesh as ASCII PLY (vertex + face elements)."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    with path.open("w", encoding="utf-8") as out:
        out.write("ply\nformat ascii 1.0\n")
        out.write(f"element vertex {v.shape[0]}\n")
        out.write("property float x\nproperty float y\nproperty float z\n")
        out.write(f"element face {f.shape[0]}\n")
        out.write("property list uchar int vertex_indices\n")
        out.write("end_header\n")
        for p in v:
            out.write(f"{p[0]} {p[1]} {p[2]}\n")
        for tri in f:
            out.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


def _parse_kf_pair(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(s).split(',') if p.strip()]
    if len(parts) != 2:
        raise ValueError("--mesh-kf-pair must be like '0,1'")
    a, b = int(parts[0]), int(parts[1])
    if a < 0 or b < 0 or a == b:
        raise ValueError("--mesh-kf-pair must contain two distinct non-negative indices")
    return a, b


def _relative_cam1_to_cam2(pose1_cw: PoseCW, pose2_cw: PoseCW) -> tuple[Mat33, Vec3]:
    """Return (R, T) mapping X2 = R X1 + T (camera-1 coords to camera-2 coords)."""
    w2 = pose2_cw.inv()  # world->cam2
    R = w2.R @ pose1_cw.R
    T = (w2.R @ pose1_cw.t) + w2.t
    return R.astype(np.float64, copy=False), T.astype(np.float64, copy=False)


def export_stereo_grid_mesh(
    ds: TempleRing,
    K: Mat33,
    kf1: Keyframe,
    kf2: Keyframe,
    cfg: StereoMeshConfig,
    out_path: Path,
    log: logging.Logger,
) -> None:
    """Create a mesh via classic stereo: rectification -> disparity -> depth -> grid triangulation.

    This is a standard, dependency-light meshing route that relies only on OpenCV + NumPy.

    Notes:
      * This uses exactly one keyframe pair.
      * Output quality depends on texture/overlap and camera geometry.
      * Scale follows the pipeline's pose scale (unless --use-gt-scale is enabled).
    """
    img1_bgr = cv2.imread(str(ds.image_path(kf1.img_name)), cv2.IMREAD_COLOR)
    img2_bgr = cv2.imread(str(ds.image_path(kf2.img_name)), cv2.IMREAD_COLOR)
    if img1_bgr is None or img2_bgr is None:
        raise FileNotFoundError('Missing keyframe image(s) for stereo mesh export')

    img1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape[:2]

    # Relative pose for OpenCV stereoRectify: cam1 -> cam2
    R, T = _relative_cam1_to_cam2(kf1.pose_cw, kf2.pose_cw)

    dist0 = np.zeros((5, 1), dtype=np.float64)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist0, K, dist0, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(K, dist0, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist0, R2, P2, (w, h), cv2.CV_32FC1)

    r1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    r2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    # StereoSGBM (classic, widely used)
    num_disp = int(cfg.num_disparities)
    if num_disp % 16 != 0:
        num_disp = int(math.ceil(num_disp / 16.0) * 16)
    block = int(cfg.block_size)
    if block % 2 == 0:
        block += 1

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block,
        P1=8 * 1 * block * block,
        P2=32 * 1 * block * block,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disp = sgbm.compute(r1, r2).astype(np.float32) / 16.0

    # Reproject disparity to 3D (rectified cam1 coordinate system)
    pts3d = cv2.reprojectImageTo3D(disp, Q).astype(np.float64)

    # Robust depth clipping in rectified frame
    z = pts3d[:, :, 2]
    z_valid = np.isfinite(z) & (disp >= float(cfg.disp_min)) & (z > 0)
    if not np.any(z_valid):
        log.warning('Stereo mesh export skipped: no valid disparity/depth')
        return
    z_cap = float(np.percentile(z[z_valid], float(cfg.z_max_percentile)))
    z_valid &= (z <= z_cap)

    step = max(1, int(cfg.step))
    ys = list(range(0, h, step))
    xs = list(range(0, w, step))

    vid = -np.ones((len(ys), len(xs)), dtype=np.int64)
    verts_rect: list[list[float]] = []

    # Create vertices
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            if not z_valid[y0, x0]:
                continue
            p = pts3d[y0, x0]
            if not np.isfinite(p).all():
                continue
            vid[yi, xi] = len(verts_rect)
            verts_rect.append([float(p[0]), float(p[1]), float(p[2])])

    if len(verts_rect) < 3:
        log.warning('Stereo mesh export skipped: insufficient valid vertices')
        return

    # Build triangles on the subsampled grid
    faces: list[list[int]] = []
    dj = float(cfg.disp_jump)

    for yi in range(len(ys) - 1):
        y0 = ys[yi]
        y1 = ys[yi + 1]
        for xi in range(len(xs) - 1):
            x0 = xs[xi]
            x1 = xs[xi + 1]

            v00 = int(vid[yi, xi])
            v01 = int(vid[yi, xi + 1])
            v10 = int(vid[yi + 1, xi])
            v11 = int(vid[yi + 1, xi + 1])

            if v00 < 0 or v01 < 0 or v10 < 0 or v11 < 0:
                continue

            d00 = float(disp[y0, x0])
            d01 = float(disp[y0, x1])
            d10 = float(disp[y1, x0])
            d11 = float(disp[y1, x1])

            # Reject quads that span large disparity jumps (helps prevent tearing across depth discontinuities)
            if (abs(d00 - d01) > dj) or (abs(d00 - d10) > dj) or (abs(d11 - d01) > dj) or (abs(d11 - d10) > dj):
                continue

            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])

    if not faces:
        log.warning('Stereo mesh export skipped: no faces survived filtering')
        return

    verts_rect_np = np.asarray(verts_rect, dtype=np.float64)

    # Convert rectified cam1 coords -> original cam1 coords via R1^T
    verts_c1 = (R1.T @ verts_rect_np.T).T

    # Original cam1 coords -> world coords via PoseCW of keyframe 1
    R_cw = kf1.pose_cw.R
    t_cw = kf1.pose_cw.t.reshape(1, 3)
    verts_w = (R_cw @ verts_c1.T).T + t_cw

    faces_np = np.asarray(faces, dtype=np.int64)

    write_ply_mesh(out_path, verts_w, faces_np)
    log.info('Wrote stereo mesh: %s (verts=%d, faces=%d, step=%d)', out_path.name, verts_w.shape[0], faces_np.shape[0], step)
def export_edges_csv(path: Path, edges: list[Edge]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "rvec_x", "rvec_y", "rvec_z", "t_x", "t_y", "t_z", "w_rot", "w_trans"])
        for e in edges:
            rvec, _ = cv2.Rodrigues(e.R_ji)
            w.writerow([e.i, e.j, float(rvec[0]), float(rvec[1]), float(rvec[2]),
                        float(e.t_ji[0]), float(e.t_ji[1]), float(e.t_ji[2]), e.w_rot, e.w_trans])


def render_input_montage(out_dir: Path, paths: list[Path]) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    target_h = 420
    resized = [im.resize((int(round(im.size[0] * (target_h / im.size[1]))), target_h)) for im in imgs]
    gap = 12
    W = sum(im.size[0] for im in resized) + gap * (len(resized) - 1)
    H = target_h + 48
    montage = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(montage)

    x = 0
    for p, im in zip(paths, resized):
        montage.paste(im, (x, 48))
        draw.text((x, 10), f"Input: {p.name}", fill=(0, 0, 0))
        x += im.size[0] + gap

    montage.save(out_dir / "input_montage.png")


def render_inlier_matches(out_dir: Path, img1: NDArray[U8], img2: NDArray[U8], pts1: Pts2, pts2: Pts2, max_draw: int = 250) -> None:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1+w2] = img2
    n = min(max_draw, pts1.shape[0])
    for i in range(n):
        x1, y1 = map(int, pts1[i])
        x2, y2 = map(int, pts2[i])
        cv2.circle(canvas, (x1, y1), 3, (0, 255, 0), -1)
        cv2.circle(canvas, (x2 + w1, y2), 3, (0, 255, 0), -1)
        cv2.line(canvas, (x1, y1), (x2 + w1, y2), (255, 0, 0), 1)
    cv2.imwrite(str(out_dir / "inlier_matches.png"), canvas)


def render_sparse_cloud(out_dir: Path, xyz: NDArray[F64]) -> None:
    if xyz.shape[0] == 0:
        return
    pts = xyz
    if pts.shape[0] > 8000:
        idx = np.random.choice(pts.shape[0], 8000, replace=False)
        pts = pts[idx]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
    ax.set_title("Sparse point cloud (triangulated tracks)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_dir / "sparse_pointcloud.png", dpi=160)
    plt.close(fig)


def render_trajectory(out_dir: Path, centers: NDArray[F64]) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(centers[:, 0], centers[:, 1], centers[:, 2])
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=20)
    ax.set_title("Keyframe camera centers")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(out_dir / "camera_trajectory.png", dpi=160)
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=Path, default=None,
                   help="Path to repo-level config.json. If omitted, uses <repo>/config.json when present.")
    ap.add_argument("--zip", type=Path, help="TempleRing zip file.")
    ap.add_argument("--extract-to", type=Path, default=Path("templeRing"), help="Extraction folder for --zip.")
    ap.add_argument("--dir", type=Path, help="Extracted dataset root containing ./templeRing/templeR_par.txt.")
    ap.add_argument("--frames", type=int, default=None, help="Number of frames to process in dataset order.")
    ap.add_argument("--out", type=Path, default=Path("outputs_klt_ba_loop_v3"), help="Output directory.")
    ap.add_argument("--use-gt-scale", action=argparse.BooleanOptionalAction, default=None,
                   help="Scale odometry using TempleRing calibrated baselines.")
    ap.add_argument("--translation-mode", choices=[m.value for m in TranslationMode], default=None,
                   help="Pose-graph constraint mode.")
    ap.add_argument("--visuals", action=argparse.BooleanOptionalAction, default=None,
                   help="Write montage/matches/trajectory/cloud images.")
    ap.add_argument("--K-yaml", type=Path, help="Optional OpenCV YAML for generic inputs (node 'K').")
    ap.add_argument("--log", default="INFO", help="Logging level.")
    ap.add_argument(
        "--export-geometry",
        choices=[m.value for m in ExportGeometry],
        default=None,
        help="Geometry export: none, pointcloud, mesh_stereo (depth-map grid), or both.",
    )
    ap.add_argument(
        "--mesh-kf-pair",
        default=None,
        help="Keyframe indices to use for stereo meshing, e.g. '0,1' (indices into keyframe list).",
    )
    ap.add_argument("--mesh-step", type=int, default=None, help="Subsample factor for the rectified grid meshing.")
    ap.add_argument("--mesh-num-disparities", type=int, default=None, help="StereoSGBM numDisparities (multiple of 16).")
    ap.add_argument("--mesh-block-size", type=int, default=None, help="StereoSGBM blockSize (odd).")
    ap.add_argument("--mesh-disp-min", type=float, default=None, help="Minimum disparity to accept as valid.")
    ap.add_argument("--mesh-disp-jump", type=float, default=None, help="Max disparity jump allowed across a triangle quad.")
    ap.add_argument("--mesh-z-max-percentile", type=float, default=None, help="Depth outlier clipping percentile in rectified frame.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log).upper(), logging.INFO), format="%(levelname)s: %(message)s")
    log = logging.getLogger("run")

    # dataset selection
    ds: TempleRing
    match (args.zip is not None, args.dir is not None):
        case (True, True):
            raise SystemExit("Provide only one of --zip or --dir.")
        case (False, False):
            raise SystemExit("Provide one of --zip or --dir.")
        case (True, False):
            ds = TempleRing.from_zip(args.zip, args.extract_to)
        case (False, True):
            ds = TempleRing.from_dir(args.dir)

    gt = {r.img: r for r in ds.records}
    K: Mat33 = ds.records[0].K if args.K_yaml is None else load_K_yaml(args.K_yaml)

    # config.json (shared across Python + C++)
    cfg_path = args.config
    if cfg_path is None:
        candidate = _default_config_path()
        cfg_path = candidate if candidate.exists() else None

    cfg_json = _load_config_json(cfg_path, log)
    common = cfg_json.get('common', {}) if isinstance(cfg_json, dict) else {}
    py_sec = cfg_json.get('python', {}) if isinstance(cfg_json, dict) else {}
    cfg_all = _deep_merge(common, py_sec)

    dflt = SystemConfig()  # built-in defaults

    frames = int(_first_not_none(args.frames, _cfg_get(cfg_all, ['system','frames'], dflt.frames)))
    use_gt_scale = bool(_first_not_none(args.use_gt_scale, _cfg_get(cfg_all, ['system','use_gt_scale'], dflt.use_gt_scale)))
    tm_s = _first_not_none(args.translation_mode, _cfg_get(cfg_all, ['system','translation_mode'], dflt.translation_mode.value))
    translation_mode = TranslationMode(str(tm_s))

    # keyframe policy
    min_inliers = int(_cfg_get(cfg_all, ['keyframe','min_inliers'], dflt.min_inliers))
    keyframe_min_gap = int(_cfg_get(cfg_all, ['keyframe','min_gap'], dflt.keyframe_min_gap))
    keyframe_parallax_px = float(_cfg_get(cfg_all, ['keyframe','parallax_px'], dflt.keyframe_parallax_px))

    loop_every_kf = int(_cfg_get(cfg_all, ['system','loop_every_kf'], dflt.loop_every_kf))
    posegraph_every_kf = int(_cfg_get(cfg_all, ['system','posegraph_every_kf'], dflt.posegraph_every_kf))

    # KLT
    klt_d = _cfg_get(cfg_all, ['klt'], {})
    if not isinstance(klt_d, dict):
        klt_d = {}
    win_size = klt_d.get('win_size', None)
    if win_size is None and 'win_radius' in klt_d:
        r = int(klt_d.get('win_radius', 5))
        win_size = [2*r + 1, 2*r + 1]
    if win_size is None:
        win_size = list(dflt.klt.win_size)

    max_level = int(klt_d.get('pyr_levels', klt_d.get('max_level', dflt.klt.max_level)))

    klt = KLTConfig(
        max_tracks=int(klt_d.get('max_tracks', dflt.klt.max_tracks)),
        min_tracks=int(klt_d.get('min_tracks', dflt.klt.min_tracks)),
        quality=float(klt_d.get('quality', dflt.klt.quality)),
        min_distance=int(klt_d.get('min_distance', dflt.klt.min_distance)),
        block_size=int(klt_d.get('block_size', dflt.klt.block_size)),
        win_size=(int(win_size[0]), int(win_size[1])),
        max_level=max_level,
        fb_thresh=float(klt_d.get('fb_thresh', dflt.klt.fb_thresh)),
    )

    # BA / loop / pose-graph
    ba_d = _cfg_get(cfg_all, ['ba'], {})
    if not isinstance(ba_d, dict):
        ba_d = {}
    ba = BAConfig(
        window=int(ba_d.get('window', dflt.ba.window)),
        max_points=int(ba_d.get('max_points', dflt.ba.max_points)),
        iters=int(ba_d.get('iters', dflt.ba.iters)),
        lambda0=float(ba_d.get('lambda0', dflt.ba.lambda0)),
        huber_delta=float(ba_d.get('huber_delta', dflt.ba.huber_delta)),
        eps_pose=float(ba_d.get('eps_pose', dflt.ba.eps_pose)),
    )

    loop_d = _cfg_get(cfg_all, ['loop_closure'], _cfg_get(cfg_all, ['loop'], {}))
    if not isinstance(loop_d, dict):
        loop_d = {}
    loop = LoopConfig(
        min_kf_gap=int(loop_d.get('min_kf_gap', dflt.loop.min_kf_gap)),
        top_k=int(loop_d.get('top_k', dflt.loop.top_k)),
        min_matches=int(loop_d.get('min_matches', dflt.loop.min_matches)),
        min_inliers=int(loop_d.get('min_inliers', dflt.loop.min_inliers)),
    )

    pg_d = _cfg_get(cfg_all, ['pose_graph'], _cfg_get(cfg_all, ['pg'], {}))
    if not isinstance(pg_d, dict):
        pg_d = {}
    pg = PoseGraphConfig(
        iters=int(pg_d.get('iters', dflt.pg.iters)),
        lambda0=float(pg_d.get('lambda0', dflt.pg.lambda0)),
        eps=float(pg_d.get('eps', dflt.pg.eps)),
        mode=translation_mode,
        w_rot=float(pg_d.get('w_rot', dflt.pg.w_rot)),
        w_trans=float(pg_d.get('w_trans', dflt.pg.w_trans)),
    )

    cfg = SystemConfig(
        frames=frames,
        use_gt_scale=use_gt_scale,
        translation_mode=translation_mode,
        min_inliers=min_inliers,
        keyframe_min_gap=keyframe_min_gap,
        keyframe_parallax_px=keyframe_parallax_px,
        loop_every_kf=loop_every_kf,
        posegraph_every_kf=posegraph_every_kf,
        klt=klt,
        ba=ba,
        loop=loop,
        pg=pg,
    )

    # output settings (shared)
    visuals = bool(_first_not_none(args.visuals, _cfg_get(cfg_all, ['outputs','visuals'], False)))
    export_geom_s = _first_not_none(args.export_geometry, _cfg_get(cfg_all, ['outputs','export_geometry'], ExportGeometry.POINTCLOUD.value))
    export_geom = ExportGeometry(str(export_geom_s))

    mesh_d = _cfg_get(cfg_all, ['mesh_stereo'], {})
    if not isinstance(mesh_d, dict):
        mesh_d = {}
    mesh_pair_s = args.mesh_kf_pair
    if mesh_pair_s is None:
        pair = mesh_d.get('kf_pair', [0, 1])
        try:
            mesh_pair_s = f"{int(pair[0])},{int(pair[1])}"
        except Exception:
            mesh_pair_s = '0,1'

    mesh_step = int(_first_not_none(args.mesh_step, mesh_d.get('step', StereoMeshConfig().step)))
    mesh_num_disp = int(_first_not_none(args.mesh_num_disparities, mesh_d.get('num_disparities', StereoMeshConfig().num_disparities)))
    mesh_block = int(_first_not_none(args.mesh_block_size, mesh_d.get('block_size', StereoMeshConfig().block_size)))
    mesh_disp_min = float(_first_not_none(args.mesh_disp_min, mesh_d.get('disp_min', StereoMeshConfig().disp_min)))
    mesh_disp_jump = float(_first_not_none(args.mesh_disp_jump, mesh_d.get('disp_jump', StereoMeshConfig().disp_jump)))
    mesh_zmax = float(_first_not_none(args.mesh_z_max_percentile, mesh_d.get('z_max_percentile', StereoMeshConfig().z_max_percentile)))

    cfg_mesh = StereoMeshConfig(
        step=mesh_step,
        num_disparities=mesh_num_disp,
        block_size=mesh_block,
        disp_min=mesh_disp_min,
        disp_jump=mesh_disp_jump,
        z_max_percentile=mesh_zmax,
    )

    sys = ClassicSystem(K, cfg, gt)

    used_paths: list[Path] = []
    for i, rec in enumerate(ds.records[: cfg.frames]):
        img_path = ds.image_path(rec.img)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Missing image: {img_path}")
        used_paths.append(img_path)
        sys.process(i, rec.img, img)
        log.info("frame %d/%d | keyframes=%d | map_points=%d | edges=%d",
                 i+1, cfg.frames, len(sys.keyframes), len(sys.map.points), len(sys.edges))

    args.out.mkdir(parents=True, exist_ok=True)

    # exports
    # Compute XYZ once (used for pointcloud export and optional visuals).
    xyz_map = sys.map.xyz()

    if export_geom in (ExportGeometry.POINTCLOUD, ExportGeometry.BOTH):
        write_ply_xyz(args.out / "templeRing_sparse_points.ply", xyz_map)

    if export_geom in (ExportGeometry.MESH_STEREO, ExportGeometry.BOTH):
        if len(sys.keyframes) < 2:
            log.warning('Stereo mesh export skipped: need at least 2 keyframes')
        else:
            a, b = _parse_kf_pair(mesh_pair_s)
            if a >= len(sys.keyframes) or b >= len(sys.keyframes):
                raise ValueError(f'mesh_stereo.kf_pair {mesh_pair_s} out of range (keyframes={len(sys.keyframes)})')
            kf1 = sys.keyframes[a]
            kf2 = sys.keyframes[b]
            out_mesh = args.out / f"templeRing_mesh_stereo_kf{a}_kf{b}.ply"
            export_stereo_grid_mesh(ds, K, kf1, kf2, cfg_mesh, out_mesh, log)

    centers = np.vstack([kf.pose_cw.t for kf in sys.keyframes]).astype(np.float64, copy=False)
    traj = pd.DataFrame({
        "kf_id": [kf.kf_id for kf in sys.keyframes],
        "frame_idx": [kf.frame_idx for kf in sys.keyframes],
        "image": [kf.img_name for kf in sys.keyframes],
        "x": centers[:, 0], "y": centers[:, 1], "z": centers[:, 2],
        "lat": [ds.angles[kf.img_name].lat for kf in sys.keyframes],
        "lon": [ds.angles[kf.img_name].lon for kf in sys.keyframes],
    })
    traj.to_csv(args.out / "keyframes_camera_centers.csv", index=False)

    export_edges_csv(args.out / "posegraph_edges.csv", sys.edges)

    # visuals
    if visuals and used_paths:
        sel = [used_paths[0], used_paths[len(used_paths)//2], used_paths[-1]]
        render_input_montage(args.out, sel)
        if sys.inlier_debug is not None:
            img1, img2, p1, p2 = sys.inlier_debug
            render_inlier_matches(args.out, img1, img2, p1, p2)
        render_sparse_cloud(args.out, xyz_map)
        render_trajectory(args.out, centers)

    print("\n=== Summary ===")
    print(f"Processed frames: {cfg.frames}")
    print(f"Keyframes:        {len(sys.keyframes)}")
    print(f"Map points:       {len(sys.map.points)}")
    print(f"Pose-graph edges: {len(sys.edges)}")
    print(f"Outputs in:       {args.out.resolve()}")


if __name__ == "__main__":
    main()
