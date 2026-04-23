"""Detector-crop + evenly-spaced pseudo-label generator (weekend-7, variant a).

Replaces `make_pseudo_labels.py`, which proposed character bboxes with the
weak synth_v2 recognizer and dropped ~64% of records to length-mismatch.
Those weak proposals also leaked into the student via distillation — a
likely cause of the synth_v3 regression.

New approach: trust the VLM class labels, don't trust any stage-2 bboxes.

  1. Run the stage-1 *plate* detector (thai-plate-ocr) to crop each image to
     the plate region.
  2. Decide 1-line vs 2-line from the crop aspect ratio (registration row
     occupies the top band on 2-line plates, full height on 1-line).
  3. Split the crop width into `len(gt_str)` equal-width slices; each slice
     becomes the bbox for one character with its class from the VLM string.

Bboxes are approximate (Thai digits are narrower than consonants), but
evenly-spaced imprecise bboxes with correct classes are a cleaner training
signal than tight bboxes with synth_v2-biased positions — and we keep
nearly every record instead of dropping 64%.

Output layout (YOLO, crop-space — matches synth dataset format):

    out_dir/
      images/train/{plate_000000.jpg, ...}
      labels/train/{plate_000000.txt, ...}
      summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from thai_plate_synth.alphabet import GLYPH_TO_CLASS

PLATE_PATTERN = re.compile(r"^[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}$")

# Aspect-ratio cutoff: crops wider than 2.5:1 are treated as single-line.
# Thai 2-line plates are typically ~2:1; single-line registration strips ~4:1+.
ASPECT_1LINE = 2.5

# Registration-row band on 2-line plates (fractions of crop height).
# Empirically tuned: registration glyphs sit roughly 5–55% of plate height.
TWO_LINE_Y_TOP = 0.05
TWO_LINE_Y_BOTTOM = 0.55

# Full-height band for 1-line crops (leave a 5% margin top/bottom).
ONE_LINE_Y_TOP = 0.05
ONE_LINE_Y_BOTTOM = 0.95

# Per-character bbox width as a fraction of the slice width.
# Slightly < 1.0 to avoid adjacent-box overlap.
CHAR_W_FRAC = 0.85


@dataclass
class DetBox:
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def _normalize(reg: str) -> str:
    return re.sub(r"[^0-9ก-ฮ]", "", reg)


def _detect_plate(det, img: Image.Image, conf: float, imgsz: int) -> DetBox | None:
    res = det.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None
    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()
    i = int(np.argmax(confs))
    return DetBox(float(confs[i]), float(xyxy[i][0]), float(xyxy[i][1]),
                  float(xyxy[i][2]), float(xyxy[i][3]))


def _even_labels(gt_str: str, crop_w: int, crop_h: int) -> list[str]:
    """Evenly-spaced YOLO bboxes across the registration band of the crop."""
    n = len(gt_str)
    aspect = crop_w / max(crop_h, 1)
    if aspect >= ASPECT_1LINE:
        y_top, y_bot = ONE_LINE_Y_TOP, ONE_LINE_Y_BOTTOM
    else:
        y_top, y_bot = TWO_LINE_Y_TOP, TWO_LINE_Y_BOTTOM
    cy = (y_top + y_bot) / 2
    bh = y_bot - y_top

    slice_w_frac = 1.0 / n
    bw = slice_w_frac * CHAR_W_FRAC
    lines = []
    for i, ch in enumerate(gt_str):
        cls = GLYPH_TO_CLASS.get(ch)
        if cls is None:
            return []
        cx = (i + 0.5) * slice_w_frac
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def make_labels(
    detector_weights: Path,
    clean_labels: Path,
    images_dir: Path,
    out_dir: Path,
    det_conf: float = 0.25,
    det_imgsz: int = 640,
) -> dict:
    from ultralytics import YOLO
    det = YOLO(str(detector_weights))

    labels = []
    for line in clean_labels.read_text().splitlines():
        if line.strip():
            labels.append(json.loads(line))

    out_images = out_dir / "images" / "train"
    out_labels = out_dir / "labels" / "train"
    out_val_img = out_dir / "images" / "val"
    out_val_lbl = out_dir / "labels" / "val"
    for d in (out_images, out_labels, out_val_img, out_val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    n_total = len(labels)
    n_no_image = 0
    n_no_detection = 0
    n_bad_reg = 0
    n_degenerate_crop = 0
    n_kept = 0

    for rec in labels:
        img_path = images_dir / rec["image"]
        if not img_path.is_file():
            n_no_image += 1
            continue

        gt_str = _normalize(rec.get("registration", ""))
        if not PLATE_PATTERN.match(gt_str):
            n_bad_reg += 1
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            n_no_image += 1
            continue

        det_box = _detect_plate(det, img, det_conf, det_imgsz)
        if det_box is None:
            # Fall back to the full image — scraped plate crops often have
            # minimal background and the detector sometimes misses them.
            crop = img
        else:
            x1, y1 = max(0, int(det_box.x1)), max(0, int(det_box.y1))
            x2, y2 = int(det_box.x2), int(det_box.y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                n_degenerate_crop += 1
                continue
            crop = img.crop((x1, y1, x2, y2))

        lines = _even_labels(gt_str, crop.width, crop.height)
        if not lines:
            continue

        stem = f"real_{n_kept:06d}"
        crop.save(out_images / f"{stem}.jpg", quality=95)
        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        n_kept += 1

        if n_kept == 1:
            crop.save(out_val_img / f"{stem}.jpg", quality=95)
            (out_val_lbl / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    summary = {
        "input_records": n_total,
        "missing_image_file": n_no_image,
        "detector_missed_falling_back_to_full_image": n_no_detection,
        "bad_registration_format": n_bad_reg,
        "degenerate_crop": n_degenerate_crop,
        "kept": n_kept,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path, default=Path(
        "/Users/zeemon/Development/thai-plate-ocr/runs/detect/artifacts/detector/train/weights/best.pt"
    ))
    ap.add_argument("--labels", type=Path, default=Path(
        "data/real_scrape/roboflow/clean_labels.jsonl"
    ))
    ap.add_argument("--images", type=Path, default=Path(
        "data/real_scrape/roboflow/images"
    ))
    ap.add_argument("--out", type=Path, default=Path("data/real_pseudo_v2"))
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-imgsz", type=int, default=640)
    args = ap.parse_args()
    s = make_labels(args.detector, args.labels, args.images, args.out,
                    args.det_conf, args.det_imgsz)
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
