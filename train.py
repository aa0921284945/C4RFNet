#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Radar–Image → LiDAR point-cloud reconstruction training script
————————————————————————————————————————————————————————————————————

Author: Luca & ChatGPT (2025-05-01)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, random, json
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.model import FusionModel, GridDecoder
from utils.dataset import RadarImageLidarDataset
from utils.loss import total_chamfer_loss
from utils.visualization import LossLogger

# ─────────────────────────── ## Configurable args ────────────────────────── #
@dataclass
class Config:
    # Data
    radar_h5:   str = "datasets/data/train_radar.h5"
    image_h5:   str = "datasets/data/train_image.h5"
    lidar_h5:   str = "datasets/data/train_lidar.h5"
    split:      float = 0.9          # train/val ratio
    batch:      int   = 16
    workers:    int   = 4

    # Model
    grid_x:     int   = 32
    grid_y:     int   = 16

    # Optimisation
    lr:         float = 3e-4
    weight_decay:float = 1e-4
    epochs:     int   = 2000
    T_max:      int   = 200         # for CosineAnnealingLR
    eta_min:    float = 3e-6
    patience:   int   = 2000
    delta:      float = 1e-4

    # Misc
    seed:       int   = 42
    log_dir:    str   = "logs"
    ckpt_dir:   str   = "models"
    smooth_k:   int   = 5           # for LossLogger.plot


def parse_cfg() -> Config:
    parser = argparse.ArgumentParser(description="Training script with CLI overrides")
    for field, typ in Config.__annotations__.items():
        default = getattr(Config, field)
        arg_type = type(default) if typ is float or typ is int or typ is str else str
        parser.add_argument(f"--{field}", type=arg_type, default=None)
    ns = parser.parse_args()
    cfg = Config()
    # override provided keys
    for k, v in vars(ns).items():
        if v is not None:
            setattr(cfg, k, v if not isinstance(getattr(cfg, k), bool) else bool(v))
    return cfg
# ──────────────────────────────────────────────────────────────────────────── #

# 0. Utilities
def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)

# 1. Data loaders
def build_loaders(cfg: Config):
    ds = RadarImageLidarDataset(cfg.radar_h5, cfg.image_h5, cfg.lidar_h5)
    keys = ds.keys.copy(); random.Random(cfg.seed).shuffle(keys)
    split_idx = int(len(keys)*cfg.split)
    id2idx = {k:i for i,k in enumerate(ds.keys)}
    tr_idx = [id2idx[k] for k in keys[:split_idx]]
    va_idx = [id2idx[k] for k in keys[split_idx:]]

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(ds, tr_idx), batch_size=cfg.batch, shuffle=True,
        num_workers=cfg.workers, worker_init_fn=seed_worker,
        pin_memory=pin, persistent_workers=cfg.workers>0, drop_last=True)
    val_loader = DataLoader(
        Subset(ds, va_idx), batch_size=cfg.batch, shuffle=False,
        num_workers=cfg.workers, worker_init_fn=seed_worker,
        pin_memory=pin, persistent_workers=cfg.workers>0, drop_last=False)

    print(f"[Data] Train {len(tr_idx)} | Val {len(va_idx)}")
    return train_loader, val_loader

# 2. Model / optimiser / scheduler
def build_components(device: torch.device, cfg: Config):
    fusion  = FusionModel().to(device)
    decoder = GridDecoder(x_size=cfg.grid_x, y_size=cfg.grid_y).to(device)
    params  = list(fusion.parameters()) + list(decoder.parameters())
    opt = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.T_max, eta_min=cfg.eta_min)
    return fusion, decoder, opt, sch

# 3. Loops
@torch.no_grad()
def evaluate(model, decoder, loader, device) -> float:
    model.eval(); decoder.eval()
    tot, n = 0.0, 0
    for radar, image, lidar, mask in loader:
        radar, image, lidar = radar.to(device), image.to(device), lidar.to(device)
        mask = mask.to(device).bool()
        pred = decoder(model(radar, image)).permute(0,2,1).contiguous()
        loss, *_ = total_chamfer_loss(pred, lidar, mask=mask)
        bs = radar.size(0); tot += loss.item()*bs; n += bs
    return tot/n

def train_one_epoch(model, decoder, loader, opt, device):
    model.train(); decoder.train()
    L_tot = G_tot = Lc_tot = 0.0
    for radar, image, lidar, mask in tqdm(loader, desc="Train", leave=False):
        radar, image, lidar = radar.to(device), image.to(device), lidar.to(device)
        mask = mask.to(device).bool()
        opt.zero_grad()
        pred = decoder(model(radar, image)).permute(0,2,1).contiguous()
        loss, g, l = total_chamfer_loss(pred, lidar, mask=mask)
        loss.backward(); opt.step()
        L_tot += loss.item(); G_tot += g.item(); Lc_tot += l.item()
    n = len(loader)
    return L_tot/n, G_tot/n, Lc_tot/n

# 4. Main
def main():
    cfg = parse_cfg()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print("[Config] " + json.dumps(asdict(cfg), indent=2))

    train_loader, val_loader = build_loaders(cfg)
    fusion, decoder, opt, sch = build_components(device, cfg)
    logger = LossLogger(cfg.log_dir)

    ckpt_dir = Path(cfg.ckpt_dir); ckpt_dir.mkdir(exist_ok=True)
    best_path = ckpt_dir / "valbest_model.pt"
    best_val, start_epoch = float("inf"), 0

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        fusion.load_state_dict(ckpt["fusion_model"])
        decoder.load_state_dict(ckpt["decoder"])
        opt.load_state_dict(ckpt["optimizer"])
        sch.load_state_dict(ckpt["scheduler"])
        best_val = ckpt["val_loss"]; start_epoch = ckpt["epoch"]
        print(f"[Resume] Epoch {start_epoch} | Best val {best_val:.4f}")

    no_imp = 0
    try:
        for epoch in range(start_epoch, cfg.epochs):
            tr, g, l = train_one_epoch(fusion, decoder, train_loader, opt, device)
            val      = evaluate(fusion, decoder, val_loader, device)

            print(f"[Ep {epoch+1:4d}] "
                  f"Train {tr:.4f} (G {g:.4f}, L {l:.4f}) | Val {val:.4f}")

            logger.log(tr, g, l, val_loss=val)

            if val + cfg.delta < best_val:
                best_val, no_imp = val, 0
                torch.save({
                    "fusion_model": fusion.state_dict(),
                    "decoder": decoder.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sch.state_dict(),
                    "epoch": epoch+1, "val_loss": val},
                    best_path)
                print(f"[Save] New best → {best_path}")
            else:
                no_imp += 1
                print(f"[EarlyStop] No improvement {no_imp}/{cfg.patience}")

            if no_imp >= cfg.patience:
                print(f"[EarlyStop] Triggered at epoch {epoch+1}")
                break

            sch.step()
    except KeyboardInterrupt:
        print("Interrupted by user.")

    # End-of-training visualisation
    logger.plot(smooth_k=cfg.smooth_k)
    logger.to_csv()
    print("Training finished.")

if __name__ == "__main__":
    main()
