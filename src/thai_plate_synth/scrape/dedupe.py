"""Perceptual-hash deduplication across the scraped corpus.

Roboflow projects frequently re-upload each other's images with different
workspace names, so cross-source dedupe is the main use case. Within-source
dupes are usually already removed by Roboflow.

We use a pHash with Hamming-distance threshold of 5 (safe default for
tolerating re-encoding + minor crops) and keep the first-seen copy per
cluster.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def _phash(img: Image.Image, hash_size: int = 8) -> int:
    """Small-footprint pHash: DCT-free, just resize + mean-threshold (a.k.a. aHash).

    Good enough for near-duplicate detection across re-encodings; cheaper
    than the true DCT-based pHash and doesn't require scipy.
    """
    g = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = list(g.tobytes())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p >= mean:
            bits |= 1 << i
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe(images_dir: Path, provenance_path: Path, *, threshold: int = 5) -> dict:
    prov: list[dict] = []
    if provenance_path.is_file():
        for line in provenance_path.read_text().splitlines():
            if line.strip():
                prov.append(json.loads(line))
    by_name = {r["image"]: r for r in prov}

    files = sorted(p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    hashes: list[tuple[Path, int]] = []
    for p in files:
        try:
            with Image.open(p) as im:
                h = _phash(im)
        except Exception:
            continue
        hashes.append((p, h))

    # Greedy cluster: for each input, compare its hash to every kept hash.
    # O(n * |keep|) which for n=5500 and |keep|≲n is ~30M ops — a few seconds.
    keep: list[Path] = []
    keep_hashes: list[int] = []
    drop: list[tuple[Path, Path, int]] = []
    for p, h in hashes:
        match_idx = -1
        for i, kh in enumerate(keep_hashes):
            if _hamming(h, kh) <= threshold:
                match_idx = i
                break
        if match_idx == -1:
            keep.append(p)
            keep_hashes.append(h)
        else:
            drop.append((p, keep[match_idx], _hamming(h, keep_hashes[match_idx])))

    # Move dropped files to a sibling _dupes/ so the operator can inspect.
    dupes_dir = images_dir.parent / "_dupes"
    dupes_dir.mkdir(exist_ok=True)
    for p, _match, _dist in drop:
        p.rename(dupes_dir / p.name)

    # Per-source tallies on the survivors.
    per_source: dict[str, int] = defaultdict(int)
    for p in keep:
        rec = by_name.get(p.name)
        if rec:
            per_source[f"{rec['source']}:{rec.get('workspace','?')}/{rec.get('project','?')}"] += 1

    # Rewrite provenance with a `kept` flag
    kept_names = {p.name for p in keep}
    new_prov = []
    for rec in prov:
        r = dict(rec)
        r["kept"] = rec["image"] in kept_names
        new_prov.append(r)
    provenance_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in new_prov) + "\n")

    return {
        "n_input": len(hashes),
        "n_kept": len(keep),
        "n_dropped": len(drop),
        "per_source_kept": dict(per_source),
        "dupes_moved_to": str(dupes_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/real_scrape/roboflow/images"))
    ap.add_argument("--provenance", type=Path, default=Path("data/real_scrape/roboflow/provenance.jsonl"))
    ap.add_argument("--threshold", type=int, default=5, help="Hamming-distance threshold (pHash64, default 5)")
    args = ap.parse_args()
    res = dedupe(args.images, args.provenance, threshold=args.threshold)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
