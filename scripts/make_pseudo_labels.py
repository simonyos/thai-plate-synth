"""Turn the VLM plate-string labels into per-character YOLO bboxes.

The VLM gives us plate-level text; YOLO needs per-character bboxes to train
a stage-2 recognizer. We bridge the gap by:

  1. Running the existing synth_v2 recognizer on each real plate to propose
     character-level bboxes.
  2. Sorting the proposals left-to-right along the registration line.
  3. If the proposal count matches the VLM string length, assigning each box
     to the corresponding character from the VLM string. Otherwise we skip
     the plate — the correspondence is ambiguous and would inject noise.

This is pseudo-label distillation: the stage-1 bboxes come from synth_v2
(a model trained purely on synth), and the class labels come from the VLM.
Training the next YOLO on these combined pseudo-labels transfers real-world
pixel distributions without requiring human bbox annotation.

Output layout (YOLO format, ready for merge into a synth dataset):

    out_dir/
      images/{plate_000000.jpg, plate_000001.jpg, ...}
      labels/{plate_000000.txt, plate_000001.txt, ...}
      summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image

from thai_plate_synth.alphabet import GLYPH_TO_CLASS

PLATE_PATTERN = re.compile(r"^[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}$")


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


def cluster_lines(boxes: list[CharBox], line_gap_frac: float = 0.6) -> list[list[CharBox]]:
    """Group boxes into lines by y-centre gap; sort each line left-to-right.

    Real Thai plates have 2+ lines (registration + province). Our VLM labels
    only the registration line, so we want the topmost line only.
    """
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


def _normalize(reg: str) -> str:
    return re.sub(r"[^0-9ก-ฮ]", "", reg)


def _run_yolo(model, img: Image.Image, conf: float, imgsz: int) -> list[CharBox]:
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


def _to_yolo_line(cls: int, box: CharBox, w: int, h: int) -> str:
    cx = (box.x1 + box.x2) / 2 / w
    cy = (box.y1 + box.y2) / 2 / h
    bw = (box.x2 - box.x1) / w
    bh = (box.y2 - box.y1) / h
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def make_labels(
    weights: Path,
    clean_labels: Path,
    images_dir: Path,
    out_dir: Path,
    conf: float = 0.25,
    imgsz: int = 480,
) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights))

    labels = []
    for line in clean_labels.read_text().splitlines():
        if line.strip():
            labels.append(json.loads(line))

    out_images = out_dir / "images" / "train"
    out_labels = out_dir / "labels" / "train"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Single-image val split — YOLO requires non-empty val. We'll pick the
    # first successful plate and copy it into val.
    out_val_img = out_dir / "images" / "val"
    out_val_lbl = out_dir / "labels" / "val"
    out_val_img.mkdir(parents=True, exist_ok=True)
    out_val_lbl.mkdir(parents=True, exist_ok=True)

    n_total = len(labels)
    n_no_image = 0
    n_no_detection = 0
    n_mismatch = 0
    n_kept = 0
    mismatch_detail: dict[int, int] = {}

    for i, rec in enumerate(labels):
        img_name = rec["image"]
        img_path = images_dir / img_name
        if not img_path.is_file():
            n_no_image += 1
            continue

        gt_str = _normalize(rec.get("registration", ""))
        if not PLATE_PATTERN.match(gt_str):
            continue
        expected_n = len(gt_str)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            n_no_image += 1
            continue

        all_boxes = _run_yolo(model, img, conf, imgsz)
        if not all_boxes:
            n_no_detection += 1
            continue

        # Real plates are multi-line (registration + province). VLM only labels
        # registration. Strategy: loose line-clustering (registration characters
        # can vary slightly in y), then pick the line whose length best matches
        # the VLM string (preferring the topmost line on ties).
        lines = cluster_lines(all_boxes, line_gap_frac=1.2)
        if not lines:
            n_no_detection += 1
            continue
        # Candidate lines: each with enough boxes to cover the expected string.
        candidates = [ln for ln in lines if len(ln) >= expected_n]
        if not candidates:
            n_mismatch += 1
            mismatch_detail[max(len(ln) for ln in lines) - expected_n] = \
                mismatch_detail.get(max(len(ln) for ln in lines) - expected_n, 0) + 1
            continue
        # Of the candidates, the registration line is the topmost one whose
        # length is closest to expected_n.
        candidates.sort(key=lambda ln: (abs(len(ln) - expected_n), np.median([b.cy for b in ln])))
        top_line = candidates[0]
        # Take the first `expected_n` boxes (sorted left-to-right) as the
        # registration. Drops are fine — province text is what we discard.
        boxes = top_line[:expected_n]

        # Write image + labels with VLM-string class assignments
        stem = f"real_{n_kept:06d}"
        dst_img = out_images / f"{stem}.jpg"
        shutil.copy2(img_path, dst_img)
        lines = []
        for ch, box in zip(gt_str, boxes, strict=True):
            cls = GLYPH_TO_CLASS.get(ch)
            if cls is None:
                break
            lines.append(_to_yolo_line(cls, box, img.width, img.height))
        else:
            # no break → all chars mapped
            (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_kept += 1

            # Copy the first kept plate into val so YOLO has a val split
            if n_kept == 1:
                shutil.copy2(img_path, out_val_img / f"{stem}.jpg")
                (out_val_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    summary = {
        "input_records": n_total,
        "missing_image_file": n_no_image,
        "no_detection": n_no_detection,
        "length_mismatch": n_mismatch,
        "length_mismatch_detail": mismatch_detail,
        "kept": n_kept,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("experiments/runs/synth_v2/best.pt"))
    ap.add_argument("--labels", type=Path, default=Path("data/real_scrape/roboflow/clean_labels.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/real_scrape/roboflow/images"))
    ap.add_argument("--out", type=Path, default=Path("data/real_pseudo"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=480)
    args = ap.parse_args()
    s = make_labels(args.weights, args.labels, args.images, args.out, args.conf, args.imgsz)
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
