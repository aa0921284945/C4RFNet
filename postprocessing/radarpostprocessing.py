#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
將 (u, v, depth, …) HDF5 反投影成雷達座標，只留 (x, y, z)。
每個 frame 固定 TARGET_N 筆 (預設 512)，不足補 0。
"""

import os, cv2, h5py
import numpy as np
from tqdm import tqdm

# === 0. 參數 ————————————————————————————————————
TARGET_N = 512                              # 固定點數
GRID_W, GRID_H = 30, 19                     # 若 (u,v) 是格點座標的範圍

# === 1. Camera intrinsics / extrinsics —————————
P2 = np.array([[1495.468642,    0.0,       961.272442, 0.0],
               [   0.0,       1495.468642, 624.89592,  0.0],
               [   0.0,          0.0,         1.0,     0.0]], dtype=np.float32)
P2_K = P2[:, :3]                            # 3×3 intrinsic

Tr_radar_to_cam = np.array(
    [[-0.013857,   -0.9997468,  0.01772762,  0.05283124],
     [ 0.10934269, -0.01913807, -0.99381983, 0.98100483],
     [ 0.99390751, -0.01183297,  0.1095802,  1.44445002]], dtype=np.float32)

Tr_cam_to_radar = np.linalg.inv(
    np.vstack([Tr_radar_to_cam, [0, 0, 0, 1]],)
)[:3, :]                                   # 3×4

# === 2. 路徑 ————————————————————————————————————
INPUT_H5  = "/home/k/C4RFNet_ws/C4RFNet/datasets/data/train_radar.h5"
OUTPUT_H5 = "/home/k/C4RFNet_ws/C4RFNet/datasets/data/train.h5"
IMAGE_DIR = "/home/k/C4RFNet_ws/C4RFNet/datasets/view_of_delft_PUBLIC/lidar/training/image_2"

# === 3. 工具函式 ——————————————————————————————
def pad_to_target(arr: np.ndarray, target_n: int) -> np.ndarray:
    """(N,D) → (target_n,D)，不足補 0，超過裁掉"""
    n, d = arr.shape
    if n >= target_n:
        return arr[:target_n]
    out = np.zeros((target_n, d), dtype=arr.dtype)
    out[:n] = arr
    return out

def grid2pixel(u, v, W, H, gw=GRID_W, gh=GRID_H):
    """30×19 網格座標 → 像素座標"""
    u_px = u * (W - 1) / (gw - 1)
    v_px = v * (H - 1) / (gh - 1)
    return u_px, v_px

def backproject_to_cam(u_px, v_px, d, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (u_px - cx) * d / fx
    y = (v_px - cy) * d / fy
    return np.stack([x, y, d], axis=1)     # (N,3)

# === 4. 主程式 ——————————————————————————————
with h5py.File(INPUT_H5, 'r') as in_h5, \
     h5py.File(OUTPUT_H5, 'w') as out_h5:

    for frame_id in tqdm(list(in_h5.keys()), desc="Grid→Radar XYZ"):
        raw = in_h5[frame_id][()]               # (512,6)

        # ★ 1) 補零過濾 -------------------------------------------------
        valid_mask = raw[:, 2] > 0              # depth>0
        if not np.any(valid_mask):
            out_h5.create_dataset(frame_id,
                                  data=np.zeros((TARGET_N, 3), np.float32))
            continue
        u, v, d = raw[valid_mask, 0], raw[valid_mask, 1], raw[valid_mask, 2]

        # ★ 2) 讀影像拿尺寸 ---------------------------------------------
        img_path = os.path.join(IMAGE_DIR, f"{frame_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        H, W = img.shape[:2]

        # ★ 3) 還原像素座標 (grid→pixel) -------------------------------
        GRID_MODE = (u.max() <= GRID_W and v.max() <= GRID_H)
        if GRID_MODE:
            u_px, v_px = grid2pixel(u, v, W, H)
        else:
            u_px, v_px = u, v

        # 4) Back-project → Camera
        cam_xyz = backproject_to_cam(u_px, v_px, d, P2_K)

        # 5) Camera → Radar
        cam_homo   = np.hstack([cam_xyz, np.ones((cam_xyz.shape[0],1), np.float32)])
        radar_xyz  = (Tr_cam_to_radar @ cam_homo.T).T[:, :3]

        # 6) 補 / 裁成 TARGET_N
        radar_xyz_fixed = pad_to_target(radar_xyz, TARGET_N)

        # 7) 寫入
        out_h5.create_dataset(frame_id, data=radar_xyz_fixed.astype(np.float32))

print(f"\n✅ 反投影完成！XYZ 已寫入：{OUTPUT_H5}")
