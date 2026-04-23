"""Compose a 3x2 sample gallery for the README. One-off helper."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Dir of plate_*.png images")
    ap.add_argument("--out", type=Path, required=True, help="Output gallery PNG")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--count", type=int, default=6)
    args = ap.parse_args()

    files = sorted(p for p in args.src.glob("plate_*.png") if "_annotated" not in p.stem)[: args.count]
    if not files:
        raise SystemExit(f"No plate_*.png under {args.src}")

    imgs = [Image.open(f).convert("RGB") for f in files]
    w, h = imgs[0].size
    pad = 16
    rows = (len(imgs) + args.cols - 1) // args.cols
    grid = Image.new("RGB", (args.cols * w + pad * (args.cols + 1), rows * h + pad * (rows + 1)), "#222")
    for i, im in enumerate(imgs):
        r, c = divmod(i, args.cols)
        grid.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"wrote {args.out} ({len(imgs)} plates)")


if __name__ == "__main__":
    main()
