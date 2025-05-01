import os, cv2, h5py
import numpy as np
from tqdm import tqdm

# === 1. Camera intrinsics / extrinsics =================================
P2 = np.array([[1495.468642, 0.0, 961.272442, 0.0],
               [0.0, 1495.468642, 624.89592, 0.0],
               [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
P2_K = P2[:, :3]                # 3×3 intrinsic

Tr_lidar_to_cam = np.array([[-0.0079802, -0.9998541,  0.0151049,  0.151],
                            [ 0.118497 , -0.0159445, -0.9928264, -0.461],
                            [ 0.9929224, -0.0061331,  0.1186069, -0.915]], dtype=np.float32)
Tr_cam_to_lidar = np.linalg.inv(np.vstack([Tr_lidar_to_cam, [0,0,0,1]]))

# === 2. Paths ===========================================================
base_dir        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_h5_path   = os.path.join(base_dir, "datasets/data/train_lidar.h5")
output_h5_path  = os.path.join(base_dir, "datasets/data/gt.h5")
image_folder    = "/home/k/C4RFNet_ws/C4RFNet/datasets/view_of_delft_PUBLIC/lidar/training/image_2"

# === 3. Helpers =========================================================
def grid2pixel(u_grid, v_grid, W, H, grid_W=30, grid_H=19):
    """把 (u,v) 從 30×19 grid 轉成影像像素座標"""
    u_px = u_grid * (W - 1) / (grid_W - 1)
    v_px = v_grid * (H - 1) / (grid_H - 1)
    return u_px, v_px

def backproject_to_3D(u_px, v_px, d, K):
    """(u,v,depth) → 相機座標系 XYZ"""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (u_px - cx) * d / fx
    y = (v_px - cy) * d / fy
    z = d
    return np.stack([x, y, z], axis=1)

# === 4. Main loop =======================================================
with h5py.File(input_h5_path, 'r') as infile, \
     h5py.File(output_h5_path, 'w') as outfile:

    for frame_id in tqdm(infile.keys(), desc="Image→LiDAR XYZ"):
        data = infile[frame_id][()]       # shape: (N,3)  u, v, depth

        # ★ 補零過濾 ---------------------------------------------------
        valid = (data[:, 2] != 0)
        if not np.any(valid):
            outfile.create_dataset(frame_id, data=np.empty((0, 3), np.float32))
            continue
        data = data[valid]

        # ★ 讀影像拿尺寸 ----------------------------------------------
        img_path = os.path.join(image_folder, f"{frame_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"找不到影像：{img_path}")
        H, W = img.shape[:2]

        # ★ 還原像素座標 (若需要) --------------------------------------
        GRID_MODE = (data[:, 0].max() <= 30 and data[:, 1].max() <= 19)
        if GRID_MODE:
            u_px, v_px = grid2pixel(data[:, 0], data[:, 1], W, H)
        else:
            u_px, v_px = data[:, 0], data[:, 1]

        # 5. Camera → LiDAR ------------------------------------------
        cam_pts        = backproject_to_3D(u_px, v_px, data[:, 2], P2_K)
        cam_pts_homo   = np.hstack([cam_pts, np.ones((cam_pts.shape[0], 1))])
        lidar_pts_homo = (Tr_cam_to_lidar @ cam_pts_homo.T).T
        lidar_xyz      = lidar_pts_homo[:, :3].astype(np.float32)

        outfile.create_dataset(frame_id, data=lidar_xyz)

print(f"\n✅ 完成！XYZ 已存到：{output_h5_path}")
