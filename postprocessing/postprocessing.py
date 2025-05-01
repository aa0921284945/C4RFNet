import os, cv2, h5py
import numpy as np
from tqdm import tqdm

# ── 1. 內參 / 外參 ─────────────────────────────
P2 = np.array([[1495.468642, 0.0, 961.272442, 0.0],
               [0.0, 1495.468642, 624.89592, 0.0],
               [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

Tr_radar_to_cam = np.array([[-0.013857, -0.9997468, 0.01772762, 0.05283124],
                            [ 0.10934269, -0.01913807,-0.99381983, 0.98100483],
                            [ 0.99390751, -0.01183297, 0.1095802,  1.44445002]], dtype=np.float32)
Tr_cam_to_radar = np.linalg.inv(np.vstack([Tr_radar_to_cam, [0,0,0,1]]))

# ── 2. 資料路徑 ────────────────────────────────
input_h5_path  = "/home/k/C4RFNet_ws/C4RFNet/datasets/data/test.h5"
output_h5_path = "/home/k/C4RFNet_ws/C4RFNet/datasets/data/final.h5"
image_folder   = "/home/k/C4RFNet_ws/C4RFNet/datasets/view_of_delft_PUBLIC/lidar/training/image_2"

# ── 3. 還原像素座標的小函式 ───────────────────
def grid2pixel(u_grid, v_grid, W, H, grid_W=30, grid_H=19):
    u_px = u_grid * (W - 1) / (grid_W - 1)
    v_px = v_grid * (H - 1) / (grid_H - 1)
    return u_px, v_px

def backproject(u_px, v_px, d, K):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (u_px - cx) * d / fx
    y = (v_px - cy) * d / fy
    return np.stack([x, y, d], axis=1)     # (N,3)

# ── 4. 主迴圈 ─────────────────────────────────
with h5py.File(input_h5_path, 'r') as infile, \
     h5py.File(output_h5_path, 'w') as outfile:

    for frame_id in tqdm(infile.keys(), desc="Grid→Radar XYZ"):
        data = infile[frame_id][()]         # (512,3)  u_grid,v_grid,depth

        # ★ 補零過濾
        valid = (data[:,2] != 0)
        if not np.any(valid):
            outfile.create_dataset(frame_id, data=np.empty((0,3), np.float32))
            continue
        data = data[valid]

        # ★ 讀影像拿尺寸
        img_path = os.path.join(image_folder, f"{frame_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        H, W = img.shape[:2]

        # ★ 還原像素座標
        u_px, v_px = grid2pixel(data[:,0], data[:,1], W, H)
        cam_pts = backproject(u_px, v_px, data[:,2], P2)

        # Camera → Radar
        cam_homo   = np.hstack([cam_pts, np.ones((cam_pts.shape[0],1))])
        radar_homo = (Tr_cam_to_radar @ cam_homo.T).T
        xyz_radar  = radar_homo[:, :3]

        outfile.create_dataset(frame_id, data=xyz_radar.astype(np.float32))

print(f"✅ 已輸出到 {output_h5_path}")
