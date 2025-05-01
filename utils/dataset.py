import h5py
import torch
from torch.utils.data import Dataset

class RadarImageLidarDataset(Dataset):
    def __init__(self, radar_h5_path, image_h5_path, lidar_h5_path):
        self.paths = dict(radar=radar_h5_path, image=image_h5_path, lidar=lidar_h5_path)
        with h5py.File(self.paths['radar'], 'r') as f:
            self.keys = sorted(list(f.keys()))
    def _open(self):
        self.radar_h5 = h5py.File(self.paths['radar'], 'r')
        self.image_h5 = h5py.File(self.paths['image'], 'r')
        self.lidar_h5 = h5py.File(self.paths['lidar'], 'r')
    def __getitem__(self, idx):
        if not hasattr(self, 'radar_h5'):
            self._open()
        k = self.keys[idx]
        radar = torch.from_numpy(self.radar_h5[k][()]).float()
        image = torch.from_numpy(self.image_h5[k][()]).float()
        lidar = torch.from_numpy(self.lidar_h5[k][()]).float()
        mask = (lidar.sum(-1)!=0).bool()
        return radar, image, lidar, mask
    def __len__(self):
        return len(self.keys)