"""FusionPointCloud Model

一次到位版：
- Encoder 全面使用 Conv1d(kernel_size=1) 實現 MLP，僅做一次 permute。
- Radar 與 Image 共用同一個 SharedEncoder。
- GridDecoder 延續 FoldingNet 思想，以 2D grid + 1×1 Conv 將 2048‑D 隱變量解展為 3D 點雲。

輸入：
    radar:  [B, N, 6]
    image:  [B, N, 5]
輸出：
    pred_pc: [B, 3, x_size*y_size] (預設 512 點)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------

class SharedEncoder(nn.Module):
    """PointNet‑style encoder implemented with 1×1 Conv (GPU‑friendly).

    Args:
        in_channels (int): feature dimension of each point (6 for radar, 5 for image).
    Returns:
        torch.Tensor: [B, 1024] global feature.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Per‑point MLP  (in → 64 → 128 → 2048)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(2048)

        # MLP after concat(64+2048=2112) → 2048 → 1024
        self.mlp1 = nn.Conv1d(2112, 2048, 1)
        self.mlp2 = nn.Conv1d(2048, 1024, 1)
        self.bn_mlp1 = nn.BatchNorm1d(2048)
        self.bn_mlp2 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, N, C]
        x = x.permute(0, 2, 1)                           # [B, C, N]
        B, _, N = x.shape

        # --- per‑point feature branch ---
        x = F.relu(self.bn1(self.conv1(x)))              # [B, 64, N]
        per_point = x                                    # [B, 64, N]

        # --- global feature branch ---
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))              # [B, 2048, N]
        global_feat = torch.max(x, dim=2, keepdim=True)[0]  # [B, 2048, 1]

        # --- concat local+global, MLP ---
        cat = torch.cat([per_point, global_feat.expand(-1, -1, N)], dim=1)  # [B, 2112, N]
        cat = F.relu(self.bn_mlp1(self.mlp1(cat)))
        cat = F.relu(self.bn_mlp2(self.mlp2(cat)))       # [B, 1024, N]

        return torch.max(cat, dim=2)[0]                  # [B, 1024]


class RadarEncoder(SharedEncoder):
    def __init__(self):
        super().__init__(in_channels=6)


class ImageEncoder(SharedEncoder):
    def __init__(self):
        super().__init__(in_channels=5)


# -----------------------------------------------------------------------------
# Fusion
# -----------------------------------------------------------------------------

class FusionModel(nn.Module):
    """Late‑fusion of radar & image global embeddings (1024 + 1024 → 2048)"""

    def __init__(self):
        super().__init__()
        self.radar_encoder = RadarEncoder()
        self.image_encoder = ImageEncoder()

    def forward(self, radar_input: torch.Tensor, image_input: torch.Tensor) -> torch.Tensor:
        radar_feat = self.radar_encoder(radar_input)   # [B, 1024]
        image_feat = self.image_encoder(image_input)   # [B, 1024]
        fused_feat = torch.cat([radar_feat, image_feat], dim=1)  # [B, 2048]
        return fused_feat


# -----------------------------------------------------------------------------
# Grid Decoder (Folding‑like)
# -----------------------------------------------------------------------------

class GridDecoder(nn.Module):
    """Decode 2048‑D fused feature to fixed‑size point cloud via 2‑stage 1×1 Conv."""

    def __init__(self, x_size: int = 32, y_size: int = 16):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.grid_points = x_size * y_size

        # stage‑1: (2048 + 2) → 256 → 64 → 3
        self.conv1 = nn.Conv1d(2050, 256, 1)
        self.conv2 = nn.Conv1d(256, 64, 1)
        self.conv3 = nn.Conv1d(64, 3, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)

        # stage‑2: (3 + 2048) → 256 → 64 → 3
        self.conv4 = nn.Conv1d(2051, 256, 1)
        self.conv5 = nn.Conv1d(256, 64, 1)
        self.conv6 = nn.Conv1d(64, 3, 1)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(64)

    # ------------------------------------------------------------------
    # helper: build normalized 2‑D grid
    # ------------------------------------------------------------------
    def build_2d_grid(self, batch_size: int, device) -> torch.Tensor:
        x = torch.linspace(0, 31, steps=self.x_size, device=device)
        y = torch.linspace(0, 15, steps=self.y_size, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")  # [y, x]
        grid = torch.stack([grid_x, grid_y], dim=0)            # [2, y, x]
        grid = grid.reshape(2, -1).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 2, P]
        return grid

    # ------------------------------------------------------------------
    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        # fused_feat: [B, 2048]
        B, _ = fused_feat.shape
        device = fused_feat.device

        # 1. make 2‑D grid  [B, 2, P]
        grid = self.build_2d_grid(B, device)

        # 2. expand fused feature to every point  [B, 2048, P]
        feat_exp = fused_feat.unsqueeze(-1).repeat(1, 1, self.grid_points)

        # 3. stage‑1
        x = torch.cat([feat_exp, grid], dim=1)           # [B, 2050, P]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        coarse = self.conv3(x)                           # [B, 3, P]

        # 4. stage‑2 refine
        x = torch.cat([coarse, feat_exp], dim=1)         # [B, 2051, P]
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        fine = self.conv6(x)                             # [B, 3, P]

        return fine   # [B, 3, P]


# -----------------------------------------------------------------------------
# Full model helper
# -----------------------------------------------------------------------------

class FusionPointCloudModel(nn.Module):
    """Helper wrapper = FusionModel + GridDecoder"""

    def __init__(self, x_size: int = 32, y_size: int = 16):
        super().__init__()
        self.fusion = FusionModel()
        self.decoder = GridDecoder(x_size, y_size)

    def forward(self, radar_pts: torch.Tensor, image_pts: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(radar_pts, image_pts)   # [B, 2048]
        pred_pc = self.decoder(fused)               # [B, 3, P]
        return pred_pc
