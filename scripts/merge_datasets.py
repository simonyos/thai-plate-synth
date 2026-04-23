"""Merge two YOLO-format datasets into one training bundle.

Used for weekend-6 distillation: combine synth_v3 (5k synth plates) with
real_pseudo (282 real plates with VLM-derived class labels + synth_v2
bbox proposals).

Produces a new dataset dir with symlinked images/labels and a fresh
dataset.yaml. Symlinks avoid a 200MB+ duplicate copy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from thai_plate_synth.alphabet import ALPHABET, N_CLASSES


def _link_split(src_dir: Path, dst_dir: Path, prefix: str, split: str) -> int:
    n = 0
    for kind in ("images", "labels"):
        src = src_dir / kind / split
        dst = dst_dir / kind / split
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            continue
        for p in src.iterdir():
            if p.is_file():
                target = dst / f"{prefix}_{p.name}"
                if target.exists():
                    target.unlink()
                os.symlink(p.resolve(), target)
                if kind == "images":
                    n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, nargs="+", required=True,
                    help="One or more YOLO dataset roots (each with images/, labels/, dataset.yaml)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    totals = {"train": 0, "val": 0}
    for i, src in enumerate(args.sources):
        prefix = src.name or f"src{i}"
        for split in ("train", "val"):
            totals[split] += _link_split(src, args.out, prefix, split)

    lines = [
        f"path: {args.out.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {N_CLASSES}",
        "names:",
    ]
    for i, g in enumerate(ALPHABET):
        lines.append(f"  {i}: {g}")
    (args.out / "dataset.yaml").write_text("\n".join(lines) + "\n")
    print(f"merged: {totals['train']} train + {totals['val']} val images → {args.out}")
    print(f"sources: {[str(s) for s in args.sources]}")


if __name__ == "__main__":
    main()
