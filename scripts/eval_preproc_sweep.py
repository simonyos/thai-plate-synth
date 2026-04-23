"""Inference-time preprocessing sweep for synth_v3b.

The weekend-8 reranker came up neutral because most remaining gold-27
errors are *not* pure consonant substitutions — they're missing/extra
characters (likely from province text bleeding into the recognizer's
context) and wrong digits.

This script tests two preprocessing knobs that address those modes
without retraining:

  1. **imgsz** at inference — synth_v3b was trained at 480; testing at
     640 and 960 trades latency for small-char resolution.
  2. **top-band re-crop** — for 2-line plates (aspect < 2.5:1), crop
     to the top 55% of the plate before stage-2, discarding province
     text. Matches the training signal (pseudo_v2 labels only span
     the top band on 2-line plates).

Outputs a sweep table + best-config per-plate view.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from eval_on_gold import _run_yolo, _string_from_boxes, _regex_filter  # noqa: E402

# Matches pseudo_v2 generator's aspect heuristic.
ASPECT_1LINE = 2.5
TWO_LINE_TOP_FRAC = 0.55  # keep top 55% of the crop height


def _top_band_crop(crop: Image.Image) -> Image.Image:
    """If the crop looks like a 2-line plate, keep only the top band."""
    if crop.width / max(crop.height, 1) >= ASPECT_1LINE:
        return crop
    new_h = int(crop.height * TWO_LINE_TOP_FRAC)
    return crop.crop((0, 0, crop.width, max(new_h, 1)))


def _lcs_acc(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    sm = SequenceMatcher(None, pred, gt, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(gt)


def eval_config(weights, det, images_dir, gold_records, imgsz: int, recrop: bool):
    rows = []
    for r in gold_records:
        img_path = images_dir / r["image"]
        if not img_path.is_file():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        det_boxes = _run_yolo(det, img, conf=0.25, imgsz=640)
        if det_boxes:
            best = max(det_boxes, key=lambda b: b.conf)
            x1, y1 = max(0, int(best.x1)), max(0, int(best.y1))
            x2, y2 = int(best.x2), int(best.y2)
            crop = img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else img
        else:
            crop = img

        if recrop:
            crop = _top_band_crop(crop)

        boxes = _run_yolo(weights, crop, conf=0.25, imgsz=imgsz)
        raw = _string_from_boxes(boxes)
        pred = _regex_filter(raw) or raw
        gt = r["gold_registration"]
        rows.append({"image": r["image"], "gt": gt, "pred": pred,
                     "acc": _lcs_acc(pred, gt)})
    n = len(rows)
    exact = sum(1 for r in rows if r["pred"] == r["gt"])
    total = sum(r["acc"] for r in rows)
    return {"n": n, "exact": exact, "char_acc": total / n if n else 0.0,
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path,
                    default=Path("experiments/runs/synth_v3b/weights/best.pt"))
    ap.add_argument("--detector", type=Path, default=Path(
        "/Users/zeemon/Development/thai-plate-ocr/runs/detect/artifacts/"
        "detector/train/weights/best.pt"
    ))
    ap.add_argument("--gold", type=Path,
                    default=Path("data/real_scrape/roboflow/gold_labels.jsonl"))
    ap.add_argument("--images", type=Path,
                    default=Path("data/real_scrape/roboflow/images"))
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/figures/gold_eval_preproc.md"))
    ap.add_argument("--imgsizes", type=int, nargs="+", default=[480, 640, 960])
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    det = YOLO(str(args.detector))

    records = []
    for line in args.gold.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("verdict") != "skip" and r.get("gold_registration"):
                records.append(r)
    print(f"evaluating on {len(records)} gold plates")

    configs = []
    for imgsz in args.imgsizes:
        for recrop in [False, True]:
            configs.append((imgsz, recrop))

    results = {}
    for imgsz, recrop in configs:
        key = (imgsz, recrop)
        print(f"  imgsz={imgsz:4d}  recrop={str(recrop):5s} …", end="", flush=True)
        res = eval_config(model, det, args.images, records, imgsz, recrop)
        print(f" n={res['n']} exact={res['exact']} char-acc={res['char_acc']:.3f}")
        results[key] = res

    md = ["# synth_v3b inference-preprocessing sweep", ""]
    md.append(f"Evaluated on {records and len(records) or 0} hand-verified readable plates.")
    md.append("Re-crop = for 2-line plates (aspect < 2.5:1), keep only the top 55% "
              "of the stage-1 crop before running stage-2.")
    md.append("")
    md.append("| imgsz | re-crop | exact | char-acc |")
    md.append("|---:|---|---:|---:|")
    for (imgsz, recrop), res in results.items():
        md.append(f"| {imgsz} | {'yes' if recrop else 'no '} | "
                  f"{res['exact']} | **{res['char_acc']:.3f}** |")
    md.append("")

    # Best-config per-plate view
    best_key = max(results, key=lambda k: results[k]["char_acc"])
    md.append(f"## Per-plate — best config (imgsz={best_key[0]}, "
              f"recrop={best_key[1]}) vs default (480, no recrop)")
    md.append("")
    md.append("| gt | default | best |")
    md.append("|---|---|---|")
    default_rows = results[(480, False)]["rows"]
    best_rows = results[best_key]["rows"]
    for d, b in zip(default_rows, best_rows):
        mark = " **↑**" if b["acc"] > d["acc"] + 1e-6 else (
            " ↓" if b["acc"] < d["acc"] - 1e-6 else "")
        md.append(f"| `{d['gt']}` | `{d['pred'] or '_'}` ({d['acc']:.2f}) | "
                  f"`{b['pred'] or '_'}` ({b['acc']:.2f}){mark} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
