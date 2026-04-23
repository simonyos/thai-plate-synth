"""Head-to-head evaluation of synth_v1 vs synth_v2 on real Thai plates.

Reuses thai-plate-ocr's stage-1 plate detector to crop plates from the 12
reproducibly-sampled validation images, then runs each synth-trained stage-2
recognizer on the crop and compares predicted strings against ground truth
(hand-labeled by the user in the prior project).

Inputs (env-configurable via argparse):
    - stage-1 detector weights (from thai-plate-ocr)
    - synth recognizer weights — both v1 (clean) and v2 (augmented)
    - 60 validation images under thai-plate-ocr/data/detector/valid/images

Outputs:
    - experiments/figures/real_eval_gallery.png — 12-image grid, v1 vs v2 labels
    - experiments/figures/real_eval_v1_v2.md — markdown comparison table
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from thai_plate_synth.alphabet import CLASS_TO_GLYPH

# Thai plate format (registration line):
#   optional 1-2 leading digits, 1-3 consonants, 1-4 digits
PLATE_PATTERN = re.compile(r"[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}")

N_IMAGES = 12
SEED = 7

# Hand-labeled ground truth (registration line only) from the prior project.
# test_02=6กพ 7414 is the one baseline example present in both; others came from user labels.
GROUND_TRUTH: dict[int, str] = {
    2: "6กพ7414",
    3: "8กย403",
    7: "5กง9640",
    9: "ฆฎ8938",
    11: "วฐ4099",
}


@dataclass(frozen=True)
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


def order_characters(chars: list[CharBox], line_gap_frac: float = 0.6) -> list[list[CharBox]]:
    if not chars:
        return []
    ordered = sorted(chars, key=lambda c: c.cy)
    median_h = float(np.median([c.height for c in ordered]))
    threshold = line_gap_frac * median_h
    lines: list[list[CharBox]] = [[ordered[0]]]
    for prev, curr in pairwise(ordered):
        if curr.cy - prev.cy > threshold:
            lines.append([curr])
        else:
            lines[-1].append(curr)
    return [sorted(ln, key=lambda c: c.cx) for ln in lines]


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


def _string_from_chars(chars: list[CharBox]) -> str:
    lines = order_characters(chars)
    return "".join(CLASS_TO_GLYPH.get(c.cls, "?") for ln in lines for c in ln)


def _plate_regex_filter(raw: str) -> str:
    """Keep the longest substring matching a valid Thai-plate registration."""
    matches = PLATE_PATTERN.findall(raw)
    return max(matches, key=len) if matches else ""


def _char_accuracy(pred: str, gt: str) -> float:
    """Longest-common-subsequence / len(gt) — forgiving of insertions."""
    if not gt:
        return 0.0
    sm = SequenceMatcher(None, pred, gt, autojunk=False)
    matches = sum(b.size for b in sm.get_matching_blocks())
    return matches / len(gt)


def _load_font(size: int):
    for path in (
        "/System/Library/Fonts/Supplemental/Ayuthaya.ttf",
        "/System/Library/Fonts/Thonburi.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=Path, default=Path(
        "/Users/zeemon/Development/thai-plate-ocr/runs/detect/artifacts/detector/train/weights/best.pt"
    ))
    ap.add_argument("--rec-v1", type=Path, default=Path("experiments/runs/synth_v1/best.pt"))
    ap.add_argument("--rec-v2", type=Path, default=Path("experiments/runs/synth_v2/best.pt"))
    ap.add_argument("--val-dir", type=Path, default=Path(
        "/Users/zeemon/Development/thai-plate-ocr/data/detector/valid/images"
    ))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/figures"))
    ap.add_argument("--conf-det", type=float, default=0.25)
    ap.add_argument("--conf-rec", type=float, default=0.25)
    args = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(str(args.detector))
    v1 = YOLO(str(args.rec_v1))
    v2 = YOLO(str(args.rec_v2))

    candidates = sorted(p for p in args.val_dir.glob("*.jpg"))
    random.seed(SEED)
    chosen = random.sample(candidates, min(N_IMAGES, len(candidates)))

    rows = []
    crops: list[Image.Image | None] = []
    for i, p in enumerate(chosen):
        img = Image.open(p).convert("RGB")
        det_chars = _run_yolo(det, img, args.conf_det, imgsz=640)
        if not det_chars:
            rows.append({"idx": i, "file": p.name, "det_conf": 0.0, "v1": "", "v2": "", "gt": GROUND_TRUTH.get(i, "")})
            crops.append(None)
            continue
        best = max(det_chars, key=lambda c: c.conf)
        x1, y1, x2, y2 = int(best.x1), int(best.y1), int(best.x2), int(best.y2)
        crop = img.crop((x1, y1, x2, y2))
        chars_v1 = _run_yolo(v1, crop, args.conf_rec, imgsz=480)
        chars_v2 = _run_yolo(v2, crop, args.conf_rec, imgsz=480)
        raw_v1 = _string_from_chars(chars_v1)
        raw_v2 = _string_from_chars(chars_v2)
        reg_v1 = _plate_regex_filter(raw_v1)
        reg_v2 = _plate_regex_filter(raw_v2)
        gt = GROUND_TRUTH.get(i, "")
        rows.append({
            "idx": i, "file": p.name, "det_conf": best.conf,
            "v1_raw": raw_v1, "v2_raw": raw_v2,
            "v1_reg": reg_v1, "v2_reg": reg_v2,
            "gt": gt,
            "v1_acc": _char_accuracy(reg_v1, gt) if gt else None,
            "v2_acc": _char_accuracy(reg_v2, gt) if gt else None,
        })
        crops.append(crop)
        gt_show = gt or "_"
        print(f"[{i:02d}] det={best.conf:.2f}  v1={reg_v1!r:<14}  v2={reg_v2!r:<18}  gt={gt_show!r}")

    # Gallery: 4x3 grid of crops with v1 (top) and v2 (bottom) labels.
    thumb_w, thumb_h = 440, 220
    cols = 3
    rows_g = (len(crops) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_w, rows_g * thumb_h), "#222")
    font = _load_font(24)
    small = _load_font(18)
    for i, (row, crop) in enumerate(zip(rows, crops, strict=True)):
        r, c = divmod(i, cols)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "black")
        if crop is not None:
            cp = crop.copy()
            cp.thumbnail((thumb_w - 20, thumb_h - 80))
            canvas.paste(cp, ((thumb_w - cp.width) // 2, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, thumb_h - 66), f"v1: {row['v1_reg'] or '_'}", fill="#9cf", font=small)
        draw.text((10, thumb_h - 42), f"v2: {row['v2_reg'] or '_'}", fill="#fc9", font=small)
        if row['gt']:
            draw.text((10, thumb_h - 22), f"gt: {row['gt']}", fill="#9f9", font=small)
        grid.paste(canvas, (c * thumb_w, r * thumb_h))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = args.out_dir / "real_eval_gallery.png"
    grid.save(gallery_path)
    print(f"\nwrote {gallery_path}")

    # Aggregate accuracy on the hand-labeled subset
    labeled = [r for r in rows if r["gt"]]
    v1_avg = sum(r["v1_acc"] for r in labeled) / len(labeled) if labeled else 0.0
    v2_avg = sum(r["v2_acc"] for r in labeled) / len(labeled) if labeled else 0.0
    v1_exact = sum(1 for r in labeled if r["v1_reg"] == r["gt"])
    v2_exact = sum(1 for r in labeled if r["v2_reg"] == r["gt"])

    md = [
        "# Real-plate eval — synth_v1 (clean) vs synth_v2 (augmented)",
        "",
        "Reproducibly sampled 12 images from the detector validation split (seed=7).",
        "`v1`/`v2` columns are the longest Thai-plate-format regex match on the",
        "raw recognizer output (i.e., the format-prior post-processing step — ",
        "item #3 in the prior project's Next Steps list).",
        "Character accuracy is LCS / |gt|, averaged across the 5 hand-labeled plates.",
        "",
        f"**synth_v1 (no aug):** char-acc = **{v1_avg:.2f}**, exact match = {v1_exact}/{len(labeled)}",
        f"**synth_v2 (augmented):** char-acc = **{v2_avg:.2f}**, exact match = {v2_exact}/{len(labeled)}",
        "",
        "| idx | file | det conf | v1 pred | v2 pred | ground truth | v1 acc | v2 acc |",
        "|---:|---|---:|---|---|---|---:|---:|",
    ]
    for r in rows:
        va = f"{r['v1_acc']:.2f}" if r['v1_acc'] is not None else "—"
        vb = f"{r['v2_acc']:.2f}" if r['v2_acc'] is not None else "—"
        md.append(
            f"| {r['idx']} | `{r['file'][:32]}…` | {r['det_conf']:.2f} | "
            f"`{r['v1_reg'] or '_'}` | `{r['v2_reg'] or '_'}` | `{r['gt'] or '_'}` | {va} | {vb} |"
        )
    md_path = args.out_dir / "real_eval_v1_v2.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")
    print(f"\nv1 char-acc = {v1_avg:.2f}  (exact: {v1_exact}/{len(labeled)})")
    print(f"v2 char-acc = {v2_avg:.2f}  (exact: {v2_exact}/{len(labeled)})")


if __name__ == "__main__":
    main()
