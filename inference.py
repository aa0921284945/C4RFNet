#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot inference & post-processing
===================================

Author : Luca & ChatGPT   Date : 2025-05-01
"""

from __future__ import annotations
from pathlib import Path
import argparse, h5py, os, cv2, numpy as np, torch
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 1. Config (change here)
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("datasets/data")
RADAR_H5   = DATA_DIR / "train_radar.h5"
IMAGE_H5   = DATA_DIR / "train_image.h5"
LIDAR_H5   = DATA_DIR / "train_lidar.h5"
IMAGE_DIR  = Path("datasets/view_of_delft_PUBLIC/"
                  "lidar/training/image_2")
CKPT_PATH  = Path("models/valbest_model.pt")
OUT_H5     = DATA_DIR / "final.h5"
BATCH_SIZE = 1
NUM_WORKER = 0
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Geometry – intrinsics / extrinsics
# ──────────────────────────────────────────────────────────────────────────────
P2 = np.array([[1495.468642, 0.0, 961.272442, 0.0],
               [0.0, 1495.468642, 624.89592, 0.0],
               [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

TR_RADAR_TO_CAM = np.array(
    [[-0.013857, -0.9997468,  0.01772762, 0.05283124],
     [ 0.10934269, -0.01913807, -0.99381983, 0.98100483],
     [ 0.99390751, -0.01183297,  0.1095802,  1.44445002]],
    dtype=np.float32
)
# Homogeneous inverse: Camera -> Radar
TR_CAM_TO_RADAR = np.linalg.inv(np.vstack([TR_RADAR_TO_CAM, [0, 0, 0, 1]]))

# ──────────────────────────────────────────────────────────────────────────────
# 3. Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def grid2pixel(u_grid: np.ndarray, v_grid: np.ndarray,
               img_w: int, img_h: int,
               grid_w: int = 30, grid_h: int = 19) -> tuple[np.ndarray, np.ndarray]:
    """Restore pixel coordinates from decoder grid indices."""
    u_px = u_grid * (img_w - 1) / (grid_w - 1)
    v_px = v_grid * (img_h - 1) / (grid_h - 1)
    return u_px, v_px


def backproject(u_px: np.ndarray, v_px: np.ndarray, depth: np.ndarray,
                K: np.ndarray) -> np.ndarray:
    """pixel + depth -> camera XYZ (N, 3)."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u_px - cx) * depth / fx
    y = (v_px - cy) * depth / fy
    return np.stack([x, y, depth], axis=1)


def grid_to_radar(grid_xyz: np.ndarray, frame_id: str) -> np.ndarray:
    """
    Convert (u_grid, v_grid, depth) predicted by GridDecoder
    back to radar XYZ coordinates.
    """
    # 1) Skip padding-zeros
    mask_valid = grid_xyz[:, 2] != 0
    if not np.any(mask_valid):
        return np.empty((0, 3), np.float32)
    grid_xyz = grid_xyz[mask_valid]

    # 2) Grab image size
    img_path = IMAGE_DIR / f"{frame_id}.jpg"
    img      = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    H, W = img.shape[:2]

    # 3) grid -> pixel -> cam
    u_px, v_px = grid2pixel(grid_xyz[:, 0], grid_xyz[:, 1], W, H)
    cam_pts    = backproject(u_px, v_px, grid_xyz[:, 2], P2)

    # 4) cam -> radar
    cam_homo   = np.hstack([cam_pts, np.ones((cam_pts.shape[0], 1))])
    radar_homo = (TR_CAM_TO_RADAR @ cam_homo.T).T
    return radar_homo[:, :3].astype(np.float32)

# ──────────────────────────────────────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    from utils.dataset import RadarImageLidarDataset
    from models.model import FusionModel, GridDecoder
    from torch.utils.data import DataLoader

    # --- dataset / dataloader ---
    test_ds = RadarImageLidarDataset(
        radar_h5_path=str(RADAR_H5),
        image_h5_path=str(IMAGE_H5),
        lidar_h5_path=str(LIDAR_H5),
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKER
    )

    # --- model ---
    fusion = FusionModel().to(DEVICE)
    decoder = GridDecoder(x_size=32, y_size=16).to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    fusion.load_state_dict(ckpt["fusion_model"])
    decoder.load_state_dict(ckpt["decoder"])
    fusion.eval(); decoder.eval()

    # --- inference + post-process ---
    OUT_H5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT_H5, "w") as h5f, torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader),
                    desc="Inference & grid→radar")
        for idx, (radar, image, _, _) in pbar:
            radar, image = radar.to(DEVICE), image.to(DEVICE)

            # (B, 3,512) → (512, 3) numpy
            pred_grid = decoder(fusion(radar, image)).permute(0, 2, 1)[0]
            pred_grid = pred_grid.cpu().numpy().astype(np.float32)

            frame_id      = test_ds.keys[idx]
            radar_xyz = grid_to_radar(pred_grid, frame_id)

            h5f.create_dataset(frame_id, data=radar_xyz)
            if idx % 200 == 0:
                pbar.set_postfix(frame=frame_id, pts=radar_xyz.shape[0])

    print(f"✅ All radar point clouds saved to {OUT_H5}")


if __name__ == "__main__":
    main()
