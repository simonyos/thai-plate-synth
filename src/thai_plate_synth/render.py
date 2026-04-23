"""Render synthetic Thai license plates with per-character YOLO bboxes.

Weekend 1 MVP — white/private single-line plates only. Format follows the
common modern Thai pattern:

    [optional-digit] [2 consonants] [4 digits]

No province line, no colour classes other than white, no geometric / lighting
augmentation yet. Those land in weekend 3.

Usage:
    uv run python -m thai_plate_synth.render --out data/synth_v1 --count 1000
    uv run python -m thai_plate_synth.render --out experiments/figures/samples --count 10 --seed 0
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from thai_plate_synth.alphabet import CONSONANTS, DIGITS, GLYPH_TO_CLASS, N_CLASSES

FONT_PATH = Path(__file__).resolve().parents[2] / "assets/fonts/SarunsThangLuang.ttf"

# Visually-similar consonant groups — each set-element is easily confused with
# at least one other element by both our YOLO recognizer and Qwen2.5-VL-3B.
# Source: gold verification of 50 real plates (see experiments/figures/
# vlm_label_audit.md). Oversampling plates composed only of these consonants
# increases the per-step gradient signal on the discriminating features.
CONFUSION_GROUPS: tuple[tuple[str, ...], ...] = (
    ("ด", "ฎ"),
    ("ต", "ถ"),
    ("พ", "ผ", "ค", "ฎ", "ม", "ฟ", "ฌ"),
    ("ร", "ธ"),
    ("ม", "ฆ"),
    ("น", "ก"),
    ("ย", "ว"),
    ("ภ", "ก"),
)
HARD_CONSONANTS: tuple[str, ...] = tuple(sorted({c for grp in CONFUSION_GROUPS for c in grp}))

# Plate canvas (white/private, single line).
PLATE_W = 440
PLATE_H = 110
FONT_SIZE = 84
TEXT_PAD_X = 16   # left/right interior margin before text starts
BORDER_PAD = 4    # thin black inner frame


@dataclass(frozen=True)
class CharAnn:
    cls: int
    glyph: str
    x1: int
    y1: int
    x2: int
    y2: int


def _sample_registration(rng: random.Random, p_hard: float = 0.0) -> list[str]:
    """Sample a plausible Thai registration string as a list of glyphs.

    With probability `p_hard`, sample consonants only from HARD_CONSONANTS
    (the visually-confusable set) to oversample discriminating examples.
    """
    n_lead = rng.choice([0, 1, 1, 2])  # weight toward 0-1 leading digits
    lead = [rng.choice(DIGITS) for _ in range(n_lead)]
    pool = HARD_CONSONANTS if rng.random() < p_hard else CONSONANTS
    cons = [rng.choice(pool) for _ in range(2)]
    tail = [rng.choice(DIGITS) for _ in range(4)]
    return lead + cons + tail


def _render_plate(glyphs: list[str], font: ImageFont.FreeTypeFont) -> tuple[Image.Image, list[CharAnn]]:
    img = Image.new("RGB", (PLATE_W, PLATE_H), "white")
    draw = ImageDraw.Draw(img)
    # thin black inner frame — matches real plates' printed border
    draw.rectangle(
        [BORDER_PAD, BORDER_PAD, PLATE_W - BORDER_PAD - 1, PLATE_H - BORDER_PAD - 1],
        outline="black",
        width=2,
    )

    # Measure each glyph to lay them out evenly, like kerned monospace.
    widths = [draw.textbbox((0, 0), g, font=font)[2] for g in glyphs]
    gap = 4
    total_w = sum(widths) + gap * (len(glyphs) - 1)
    x = (PLATE_W - total_w) // 2
    # Visual centring: ThangLuang glyphs have consistent cap-height, so align by baseline.
    ascent, descent = font.getmetrics()
    y = (PLATE_H - (ascent + descent)) // 2

    anns: list[CharAnn] = []
    for g, w in zip(glyphs, widths, strict=True):
        draw.text((x, y), g, fill="black", font=font)
        # Tight bbox via textbbox around the drawn position
        tb = draw.textbbox((x, y), g, font=font)
        anns.append(
            CharAnn(
                cls=GLYPH_TO_CLASS[g],
                glyph=g,
                x1=tb[0],
                y1=tb[1],
                x2=tb[2],
                y2=tb[3],
            )
        )
        x += w + gap

    return img, anns


def _yolo_lines(anns: list[CharAnn], w: int, h: int) -> list[str]:
    out = []
    for a in anns:
        cx = (a.x1 + a.x2) / 2 / w
        cy = (a.y1 + a.y2) / 2 / h
        bw = (a.x2 - a.x1) / w
        bh = (a.y2 - a.y1) / h
        out.append(f"{a.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return out


def _write_dataset_yaml(out_dir: Path) -> None:
    from thai_plate_synth.alphabet import ALPHABET

    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {N_CLASSES}",
        "names:",
    ]
    for i, g in enumerate(ALPHABET):
        lines.append(f"  {i}: {g}")
    (out_dir / "dataset.yaml").write_text("\n".join(lines) + "\n")


def generate(
    out: Path,
    count: int,
    seed: int,
    val_frac: float = 0.1,
    annotated_preview: bool = False,
    aug: bool = False,
    p_hard: float = 0.0,
) -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(
            f"Font not found at {FONT_PATH}. See assets/fonts/README.md for the download link."
        )
    font = ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    rng = random.Random(seed)

    if aug:
        # Local import so non-aug runs don't depend on numpy indirectly.
        from thai_plate_synth.augment import AugConfig, apply as apply_aug
        aug_cfg = AugConfig()
    else:
        apply_aug = None
        aug_cfg = None

    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(count * val_frac)) if count > 1 else 0
    n_train = count - n_val

    def _emit(idx: int, split: str) -> None:
        glyphs = _sample_registration(rng, p_hard=p_hard)
        img, anns = _render_plate(glyphs, font)
        if apply_aug is not None:
            img, anns = apply_aug(img, anns, rng, aug_cfg)
            if not anns:
                # All bboxes warped off-canvas; skip (extremely rare at default settings).
                return
        name = f"plate_{idx:06d}"
        img.save(out / "images" / split / f"{name}.png")
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(_yolo_lines(anns, img.width, img.height)) + "\n"
        )
        if annotated_preview:
            preview = img.copy()
            d = ImageDraw.Draw(preview)
            for a in anns:
                d.rectangle([a.x1, a.y1, a.x2, a.y2], outline="red", width=2)
            preview.save(out / "images" / split / f"{name}_annotated.png")

    for i in range(n_train):
        _emit(i, "train")
    for i in range(n_val):
        _emit(n_train + i, "val")

    _write_dataset_yaml(out)
    print(f"Wrote {n_train} train + {n_val} val plates to {out} (aug={aug}, p_hard={p_hard})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render synthetic Thai license plates.")
    ap.add_argument("--out", type=Path, required=True, help="Output dataset root")
    ap.add_argument("--count", type=int, default=1000, help="Total plates (train+val)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Fraction for val split")
    ap.add_argument("--annotated", action="store_true", help="Also save per-char bbox overlays")
    ap.add_argument("--aug", action="store_true", help="Apply photometric/geometric augmentation")
    ap.add_argument(
        "--p-hard", type=float, default=0.0,
        help="Prob [0,1] of sampling consonants from the confusion-prone set; see CONFUSION_GROUPS",
    )
    args = ap.parse_args()
    generate(args.out, args.count, args.seed, args.val_frac, args.annotated, args.aug, args.p_hard)


if __name__ == "__main__":
    main()
