"""Evaluate synth recognizer weights against the hand-verified gold set.

Ground truth comes from `data/real_scrape/roboflow/gold_labels.jsonl` —
the 23 hand-verified readable plates. Images are in the scraped corpus
(registered in `provenance.jsonl`).

We evaluate TWO modes side by side for each weight:
  a. Recognizer-only: feed the pre-cropped plate image directly.
  b. Detector → crop → recognizer: chain with the thai-plate-ocr stage-1
     detector, to isolate the gain from pure stage-2 improvement.

Outputs a markdown comparison table + per-weight char accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from thai_plate_synth.alphabet import CLASS_TO_GLYPH

PLATE_PATTERN = re.compile(r"[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}")


@dataclass
class CharBox:
    cls: int
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def height(self) -> float: return self.y2 - self.y1


def order_chars(boxes: list[CharBox], line_gap_frac: float = 0.6) -> list[list[CharBox]]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: b.cy)
    median_h = float(np.median([b.height for b in ordered]))
    threshold = line_gap_frac * median_h
    lines: list[list[CharBox]] = [[ordered[0]]]
    for prev, curr in pairwise(ordered):
        if curr.cy - prev.cy > threshold:
            lines.append([curr])
        else:
            lines[-1].append(curr)
    return [sorted(ln, key=lambda b: b.cx) for ln in lines]


def _run_yolo(model, img, conf: float, imgsz: int) -> list[CharBox]:
    res = model.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    return [
        CharBox(int(cid), float(cv), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        for cid, cv, b in zip(cls_ids, confs, xyxy, strict=False)
    ]


def _string_from_boxes(boxes: list[CharBox]) -> str:
    lines = order_chars(boxes)
    if not lines:
        return ""
    # Take the topmost line — registration for multi-line plates.
    return "".join(CLASS_TO_GLYPH.get(c.cls, "?") for c in lines[0])


def _regex_filter(s: str) -> str:
    matches = PLATE_PATTERN.findall(s)
    return max(matches, key=len) if matches else ""


def _lcs_acc(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    sm = SequenceMatcher(None, pred, gt, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(gt)


def eval_weights(weights_path: Path, images_dir: Path, gold_records: list[dict],
                 detector_weights: Path | None = None,
                 conf: float = 0.25, imgsz: int = 480) -> dict:
    """If a detector is provided, chain: stage-1 detector → crop → stage-2.

    Scraped images are a mix of scene photos and pre-cropped plates. For scene
    images, a stage-1 crop is essential (the recognizer was trained on crops).
    For pre-cropped plates, the detector just returns a near-full-image box,
    which is harmless.
    """
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    det = YOLO(str(detector_weights)) if detector_weights else None

    n = 0
    exact = 0
    total_acc = 0.0
    rows = []
    for r in gold_records:
        img_path = images_dir / r["image"]
        if not img_path.is_file():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        crop = img
        if det is not None:
            det_boxes = _run_yolo(det, img, conf=0.25, imgsz=640)
            if det_boxes:
                best = max(det_boxes, key=lambda b: b.conf)
                x1, y1, x2, y2 = int(best.x1), int(best.y1), int(best.x2), int(best.y2)
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))

        boxes = _run_yolo(model, crop, conf, imgsz)
        raw = _string_from_boxes(boxes)
        pred = _regex_filter(raw) or raw
        gt = r["gold_registration"]
        acc = _lcs_acc(pred, gt)
        total_acc += acc
        if pred == gt:
            exact += 1
        n += 1
        rows.append({"image": r["image"], "gt": gt, "pred": pred, "raw": raw, "acc": acc})

    return {
        "weights": str(weights_path),
        "n": n,
        "exact_match": exact,
        "exact_rate": exact / n if n else 0.0,
        "char_acc": total_acc / n if n else 0.0,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, nargs="+", required=True,
                    help="One or more YOLO weights files to compare")
    ap.add_argument("--detector", type=Path, default=Path(
        "/Users/zeemon/Development/thai-plate-ocr/runs/detect/artifacts/detector/train/weights/best.pt"
    ), help="Stage-1 plate detector (from thai-plate-ocr); set to '' to skip")
    ap.add_argument("--gold", type=Path, default=Path("data/real_scrape/roboflow/gold_labels.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/real_scrape/roboflow/images"))
    ap.add_argument("--out", type=Path, default=Path("experiments/figures/gold_eval.md"))
    args = ap.parse_args()
    det_path = args.detector if args.detector and str(args.detector) != "" and Path(args.detector).is_file() else None
    if det_path is None:
        print("WARN: no stage-1 detector found; running recognizer directly on source images")

    # Load gold records, skip unreadable (verdict == 'skip')
    records = []
    for line in args.gold.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("verdict") != "skip" and r.get("gold_registration"):
                records.append(r)
    print(f"evaluating against {len(records)} readable gold plates")

    results = []
    for w in args.weights:
        print(f"  {w}:")
        res = eval_weights(w, args.images, records, detector_weights=det_path)
        results.append(res)
        print(f"    n={res['n']}  exact={res['exact_match']}/{res['n']} ({res['exact_rate']:.1%})  char-acc={res['char_acc']:.3f}")

    def _name(p: Path) -> str:
        """runs/<name>/best.pt → <name>; else fall back."""
        parts = p.parts
        # find runs/<name>/weights/best.pt  OR  runs/<name>/best.pt
        for i, part in enumerate(parts):
            if part == "runs" and i + 1 < len(parts):
                return parts[i + 1]
        return p.stem

    # Write markdown
    md = ["# Gold-set eval — synth recognizer versions", ""]
    md.append(f"Evaluated on {records and len(records) or 0} hand-verified readable plates from the Roboflow-scraped corpus.")
    md.append("Char-accuracy is LCS / |gt|. Regex post-filter applied to recognizer output.")
    md.append("")
    md.append("| weights | n | exact | exact rate | char-acc |")
    md.append("|---|---:|---:|---:|---:|")
    for res in results:
        md.append(f"| `{_name(Path(res['weights']))}` | {res['n']} | {res['exact_match']} | {res['exact_rate']:.1%} | **{res['char_acc']:.3f}** |")
    md.append("")

    # Per-plate side-by-side across all weights
    if results:
        names = [_name(Path(r['weights'])) for r in results]
        md.append("## Per-plate predictions (side-by-side)")
        md.append("")
        md.append(f"| gt | {' | '.join(names)} |")
        md.append(f"|---|{'|'.join(['---'] * len(names))}|")
        n_rows = len(results[0]["rows"])
        for ri in range(n_rows):
            cells = [f"`{results[0]['rows'][ri]['gt']}`"]
            for res in results:
                row = res["rows"][ri]
                cells.append(f"`{row['pred'] or '_'}` ({row['acc']:.2f})")
            md.append("| " + " | ".join(cells) + " |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
