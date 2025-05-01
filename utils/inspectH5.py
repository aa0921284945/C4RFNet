import h5py
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Script to inspect the structure of an HDF5 file (e.g., train_radar.h5)
# - Lists all dataset keys
# - Prints shape and preview of first 5 entries
# ---------------------------------------------------------------------

this_dir = Path(__file__).resolve().parent
h5_path = this_dir.parent / "datasets/data/train_lidar.h5"

with h5py.File(h5_path, "r") as f:
    keys = sorted(f.keys())
    total = len(keys)

    print(f"🔎 Found {total} keys in file: {h5_path.name}")

    if total == 0:
        print("⚠️  This file contains no datasets.")
    else:
        first_five = keys[:5]
        last_five  = keys[-5:] if total >= 5 else keys

        print("\n--- First 5 keys ---")
        for k in first_five:
            print(k)

        print("\n--- Last 5 keys ---")
        for k in last_five:
            print(k)

        print("\n=== Details for first 5 keys ===")
        for i, key in enumerate(first_five, start=1):
            data = f[key][:]
            print(f"\n[{i}] Key: {key}")
            print(f"   shape: {data.shape}, dtype: {data.dtype}")
            print(f"   First 5 rows:\n{data[:5]}")
