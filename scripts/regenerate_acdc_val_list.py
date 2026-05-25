#!/usr/bin/env python3
"""Regenerate ACDC val.list to include only existing volume HDF5 files.

Usage:
  python scripts/regenerate_acdc_val_list.py --acdc_dir /content/KnowSAM/ACDC

If an existing `val.list` is present, the script will filter it to keep only
names that have a corresponding file in `acdc_dir/data/*.h5`.
If no `val.list` exists, it will write all found `.h5` basenames to `val.list`.
"""
import os
import argparse


def main(acdc_dir: str):
    data_dir = os.path.join(acdc_dir, 'data')
    if not os.path.isdir(acdc_dir):
        raise SystemExit(f"ACDC dir not found: {acdc_dir}")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"ACDC data dir not found: {data_dir}")

    h5_files = sorted([os.path.splitext(os.path.basename(p))[0] for p in
                       [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
                       if p.lower().endswith('.h5')])

    val_list_path = os.path.join(acdc_dir, 'val.list')
    if os.path.exists(val_list_path):
        with open(val_list_path, 'r') as f:
            original = [l.strip() for l in f if l.strip()]
        kept = [n for n in original if n in h5_files]
        missing = [n for n in original if n not in h5_files]
        print(f"original val.list: {len(original)} entries; found {len(kept)} available, {len(missing)} missing")
        if missing:
            print("examples missing:", missing[:10])
        to_write = kept
    else:
        print(f"No existing val.list found; writing {len(h5_files)} entries from {data_dir}")
        to_write = h5_files

    with open(val_list_path, 'w') as f:
        for name in to_write:
            f.write(name + '\n')

    print(f"Wrote {len(to_write)} entries to {val_list_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acdc_dir', required=True, help='Path to ACDC folder (contains data/ and val.list)')
    args = parser.parse_args()
    main(args.acdc_dir)
