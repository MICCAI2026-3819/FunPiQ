import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from badish.dataset.sif_dataset import SIFDataset


def prepare_patch_dataset(sif_dataset: SIFDataset):
    X_all = []
    y_all = []
    for _, X, M in tqdm(sif_dataset, desc="Preparing patch dataset"):
        h, w, d = X.shape
        X_all.append(X.reshape(h * w, d))
        y_all.append(M.reshape(h * w))

    X_all = np.concatenate(X_all, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(y_all, axis=0).astype(np.uint8, copy=False)
    return X_all, y_all


def parse_args():
    parser = argparse.ArgumentParser(description="Build a patch-level BADISH dataset from feature, mask, and image folders.")
    parser.add_argument("--feature-dir", required=True, type=Path)
    parser.add_argument("--seg-dir", required=True, type=Path)
    parser.add_argument("--img-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = SIFDataset(args.feature_dir, args.seg_dir, args.img_dir)
    X, y = prepare_patch_dataset(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X, y=y)
    print(f"Saved {len(y)} patches with feature dim {X.shape[1]} to {args.output}")


if __name__ == "__main__":
    main()
