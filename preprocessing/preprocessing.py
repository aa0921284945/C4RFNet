#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build three HDF5 datasets aligned on the image plane:

    1. LiDAR  → image-plane  ([u, v, depth]           , 512 pts)
    2. Radar  → image-plane  ([u, v, depth, RCS, vr…], 512 pts)
    3. Image  → pseudo points([row, col, R, G, B]    , 570 pts)

Compared with the original script:
    • Removed repeated code via generic helpers.
    • Switched to pathlib for clearer path handling.
    • Eliminated many disk seeks (re-use image size instead of re-reading file).
    • Added argparse so you can override paths & parameters from CLI.
    • Added rich type hints and stricter error handling.
    • All comments are now in English.

Author: Luca & ChatGPT (2025-05-01)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch_geometric.nn import fps
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                      ---------- CONFIGURABLE CONSTANTS ----------            #
# --------------------------------------------------------------------------- #
TARGET_H, TARGET_W = 19, 30    # Down-sampled image grid   (height, width)
TARGET_N           = 512       # Fixed #points per frame (LiDAR/Radar)
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
#                           ---------- UTILITIES ----------                    #
# --------------------------------------------------------------------------- #
def _pad(arr: np.ndarray, n: int) -> np.ndarray:
    """Pad `arr` (m×d) with zero rows so its first dim equals `n`."""
    m, d = arr.shape
    if m >= n:
        return arr
    out = np.zeros((n, d), dtype=arr.dtype)
    out[:m] = arr
    return out


def _scale_uv(uv: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Scale raw pixel (u,v) → down-sampled grid coordinate."""
    u = np.clip(uv[:, 0], 0, img_w - 1) / (img_w - 1) * (TARGET_W - 1)
    v = np.clip(uv[:, 1], 0, img_h - 1) / (img_h - 1) * (TARGET_H - 1)
    return np.stack([u, v], axis=1)


def _fps_idx(xyz: np.ndarray, n: int) -> np.ndarray:
    """Return ≤n indices by farthest-point-sampling."""
    if xyz.shape[0] <= n:
        return np.arange(xyz.shape[0])
    ratio = n / xyz.shape[0]
    idx = fps(
        torch.as_tensor(xyz, dtype=torch.float32, device=DEVICE),
        batch=torch.zeros(len(xyz), dtype=torch.long, device=DEVICE),
        ratio=ratio,
        random_start=True,
    ).cpu().numpy()
    if idx.size > n:                          # Very rarely >n
        idx = np.random.choice(idx, n, replace=False)
    return idx


def _fov_mask(xyz: np.ndarray,
              *,
              h_deg: float,
              v_deg: float,
              max_r: float) -> np.ndarray:
    """Boolean mask for a simple conical FOV."""
    x, y, z = xyz.T
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    h_ang = np.degrees(np.arctan2(y, x))
    v_ang = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))
    return (
        (dist <= max_r) &
        (np.abs(h_ang) <= h_deg) &
        (np.abs(v_ang) <= v_deg)
    )


def _project(points_cam: np.ndarray,
             P2: np.ndarray) -> np.ndarray:
    """Project 3-D points (cam frame) → homogeneous pixel coord (N×3)."""
    proj = (P2 @ np.hstack([points_cam, np.ones((len(points_cam), 1))]).T).T
    proj[:, :2] /= proj[:, 2:3]
    return proj


# --------------------------------------------------------------------------- #
#                      ---------- GENERIC PIPELINE ----------                  #
# --------------------------------------------------------------------------- #
def _process_cloud(
    cloud_paths: list[Path],
    img_shape: tuple[int, int],
    P2: np.ndarray,
    Tr: np.ndarray,
    out_path: Path,
    *,
    fields_per_point: int,
    h_deg: float,
    v_deg: float,
    max_r: float,
    keep_extras: int = 3,     # NEW: how many extra channels to keep (0–3)
) -> None:
    """Core routine for LiDAR and Radar."""
    keep_extras = max(0, min(keep_extras, 3))       # clamp
    h5 = h5py.File(out_path, "w")

    img_h, img_w = img_shape
    for path in tqdm(cloud_paths,
                     total=len(cloud_paths),
                     desc=f"→ {out_path.name}"):
        key = path.stem
        raw = np.fromfile(path, dtype=np.float32).reshape(-1, 3 + fields_per_point)
        pts, extras = raw[:, :3], raw[:, 3:]

        # 1) FOV filter
        mask = _fov_mask(pts, h_deg=h_deg, v_deg=v_deg, max_r=max_r)
        pts, extras = pts[mask], extras[mask]
        if len(pts) == 0:
            h5.create_dataset(key, data=np.zeros((TARGET_N, 3 + keep_extras)))
            continue

        # 2) Sensor → cam frame
        cam = (Tr @ np.c_[pts, np.ones(len(pts))].T).T
        front = cam[:, 2] > 0
        cam, extras = cam[front], extras[front]
        if len(cam) == 0:
            h5.create_dataset(key, data=np.zeros((TARGET_N, 3 + keep_extras)))
            continue

        # 3) FPS
        idx = _fps_idx(cam[:, :3], TARGET_N)
        cam, extras = cam[idx], extras[idx]

        # 4) Project → image plane & crop inside grid
        proj = _project(cam[:, :3], P2)            # N×3
        uv = _scale_uv(proj[:, :2], img_w, img_h)
        inside = (
            (uv[:, 0] >= 0) & (uv[:, 0] < TARGET_W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < TARGET_H)
        )

        depth = proj[inside, 2:3]                  # N×1
        if keep_extras > 0:
            extras_kept = extras[inside, :keep_extras]
            final = np.hstack([uv[inside], depth, extras_kept])
        else:
            final = np.hstack([uv[inside], depth])

        final = _pad(final, TARGET_N).astype(np.float32)
        h5.create_dataset(key, data=final)
    h5.close()


def build_image_pseudo_h5(
    image_paths: list[Path],
    out_path: Path,
    show_preview: bool = False,
) -> None:
    """Down-sample each RGB image to TARGET_W × TARGET_H and save pseudo-points."""
    with h5py.File(out_path, "w") as h5:
        for img_path in tqdm(image_paths, desc=f"→ {out_path.name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (TARGET_W, TARGET_H),
                             interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape
            rows, cols = np.meshgrid(
                np.arange(h, dtype=np.float32),
                np.arange(w, dtype=np.float32),
                indexing="ij",
            )
            pts = np.stack(
                [rows.ravel(), cols.ravel(),
                 img[:, :, 0].ravel(),
                 img[:, :, 1].ravel(),
                 img[:, :, 2].ravel()],
                axis=1,
            )
            h5.create_dataset(img_path.stem, data=pts)

            if show_preview:
                cv2.imshow("preview", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == 27:   # ESC
                    show_preview = False
    if show_preview:
        cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
#                                 ----- MAIN -----                            #
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate LiDAR / Radar / Image projection datasets."
    )
    ap.add_argument("--base", type=Path,
                    default=Path(__file__).resolve().parent / "..",
                    help="Project root containing the datasets/ folder")
    args = ap.parse_args()

    # ── Paths & camera parameters (EDIT as needed) ────────────────────────── #
    base = args.base.resolve()
    lidar_dir  = base / "datasets/view_of_delft_PUBLIC/lidar/training/velodyne"
    radar_dir  = base / "datasets/view_of_delft_PUBLIC/radar/training/velodyne"
    image_dir  = base / "datasets/view_of_delft_PUBLIC/lidar/training/image_2"

    out_dir    = base / "datasets/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_lidar  = out_dir / "train_lidar.h5"
    out_radar  = out_dir / "train_radar.h5"
    out_image  = out_dir / "train_image.h5"

    P2 = np.array([[1495.468642, 0., 961.272442, 0.],
                   [0., 1495.468642, 624.89592, 0.],
                   [0., 0., 1., 0.]], dtype=np.float32)

    Tr_lidar2cam = np.array(
        [[-0.0079802, -0.9998541, 0.0151049, 0.151],
         [0.118497, -0.0159445, -0.9928264, -0.461],
         [0.9929224, -0.0061331, 0.1186069, -0.915]],
        dtype=np.float32,
    )

    Tr_radar2cam = np.array(
        [[-0.013857, -0.9997468, 0.01772762, 0.05283124],
         [0.10934269, -0.01913807, -0.99381983, 0.98100483],
         [0.99390751, -0.01183297, 0.1095802, 1.44445002]],
        dtype=np.float32,
    )

    # ── Image size (read only once!) ──────────────────────────────────────── #
    sample_img = next(image_dir.glob("*.jpg"), None)
    if sample_img is None:
        sys.exit("❌ No images found.")
    img_h, img_w = cv2.imread(str(sample_img)).shape[:2]

    # ── Build datasets ────────────────────────────────────────────────────── #
    _process_cloud(
        list(sorted(lidar_dir.glob("*.bin"))),
        (img_h, img_w),
        P2, Tr_lidar2cam, out_lidar,
        fields_per_point=1,          # LiDAR .bin: x y z i
        h_deg=30, v_deg=20, max_r=150,
        keep_extras=0,               # ← store only u v depth (3 dims)
    )

    _process_cloud(
        list(sorted(radar_dir.glob("*.bin"))),
        (img_h, img_w),
        P2, Tr_radar2cam, out_radar,
        fields_per_point=4,          # Radar .bin: x y z RCS vr vr_c
        h_deg=30, v_deg=20, max_r=150,
        keep_extras=3,               # keep RCS, vr, vr_c
    )

    build_image_pseudo_h5(
        list(sorted(image_dir.glob("*.jpg"))),
        out_image,
        show_preview=False,
    )

    print("✅ Projection datasets generated.")


if __name__ == "__main__":
    main()
