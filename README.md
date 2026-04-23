# Thai Plate Synth — Synthetic Data for Thai License-Plate OCR

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simonyos/thai-plate-synth/blob/main/notebooks/thai_plate_synth_colab.ipynb)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end Thai license-plate OCR with a **synthetic-data-first** pipeline.
The research asset is an open plate renderer that covers the full Thai alphabet;
the product asset is a YOLOv8 recognizer and FastAPI/Streamlit demo trained on it.

## Why

Public Thai-plate datasets are small, partially anonymised (opaque `A##` class
labels), and don't cover all 44 Thai consonants or province text. Our prior
project [`thai-plate-ocr`](https://github.com/simonyos/thai-plate-ocr) ran into
exactly this ceiling: only 8 of 44 consonants could be mapped from the public
labels. Hand-labeling at scale is expensive.

This project tests whether a parametric plate renderer — using the canonical
Thai highway-signage font and domain-randomised augmentation — can replace
most of that human labeling effort.

## Thesis

> Synthetic plates rendered with the canonical font plus realistic augmentation
> can train a stage-2 character recognizer that matches or exceeds the
> real-data baseline, with **<2 hours** of human labeling needed only for a
> held-out real-world test set.

## Plan

| Weekend | Deliverable |
|---|---|
| 1 | Renderer MVP — white/private plates, all 44 consonants + 10 digits, YOLO char bboxes |
| 2 | Train stage-2 on synth-only; evaluate on the `thai-plate-ocr` validation gallery |
| 3 | Realism pass — perspective, motion blur, lighting, background compositing |
| 4 | Hand-label ~50 real plates as held-out benchmark; run all training regimes |
| 5 | Streamlit demo + Hugging Face Space + writeup |
| 6 | Polish, province rendering, demo GIF |

## Setup

```bash
# 1. Download the font (not redistributable — see assets/fonts/README.md)
#    https://www.f0nt.com/release/saruns-thangluang/  →  assets/fonts/SarunsThangLuang.ttf

# 2. Install
make setup

# 3. Generate a sample
make sample       # 10 plates → experiments/figures/samples/
make synth        # 1000 plates → data/synth_v1/
```

## Sample renders

**Clean (weekend-1):**

![Clean gallery](experiments/figures/synth_gallery.png)

**Augmented (weekend-3 — perspective, photometric, blur, noise, JPEG):**

![Augmented gallery](experiments/figures/synth_gallery_aug.png)

All 44 Thai consonants and 10 digits are in the class space; per-character
YOLO bboxes ship alongside every image and are re-projected through each
geometric augmentation.

## Training runs

YOLOv8n, 50 epochs, imgsz 480, batch 64, seed 42, 5,000 plates each
(4,500 train / 500 val), RTX 3060 Ti 8 GB.

| Run | Augmentation | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Train time |
|---|---|---:|---:|---:|---:|---:|
| `synth_v1` | none | 0.995 | **0.995** | 0.997 | 1.000 | 13 min |
| `synth_v2` | perspective + photometric + blur + noise + JPEG | 0.995 | **0.987** | 0.996 | 0.998 | 12.5 min |

The augmented run drops mAP@0.5:0.95 by 0.8 points — exactly the price we
pay for training on perspective-warped boxes whose edges are inherently
less tight. mAP@0.5 and per-class accuracy are unchanged. That's the shape
we want: the model generalises over viewing angles and sensor noise
without losing the clean-case performance.

**Training curves, synth_v2:**

![Training curves](experiments/figures/synth_v2/results.png)

**Confusion matrix, synth_v2** — still clean-diagonal across all 54 classes:

![Confusion matrix](experiments/figures/synth_v2/confusion_matrix.png)

## Weekend 4 — head-to-head on real plates

Stage-1 detector from the prior [`thai-plate-ocr`](https://github.com/simonyos/thai-plate-ocr)
project crops plates from 12 reproducibly-sampled validation images (seed=7);
both synth-trained stage-2 weights are then run on the crops, post-processed
with a Thai-plate regex (`[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}`), and graded against
the 5 plates hand-labeled in the prior project.

**Accuracy** (longest-common-subsequence / len(gt), averaged over 5 labeled plates):

| Run | Augmentation | Char-accuracy on real plates | Exact match |
|---|---|---:|---:|
| `synth_v1` | none | 0.15 | 0/5 |
| `synth_v2` | full pipeline | **0.66** | 0/5 |

Augmentation lifts character accuracy **4.4×** on real plates — from basically
unusable (0.15) to reading most of the registration line (0.66). The
remaining errors on `synth_v2` are consonant substitutions (ฟ↔พ, ฮ↔8, ฉ↔ฆ)
and a missing digit on two plates. No plate hits exact match yet; the
consonant-confusion set is the next bottleneck.

![Real-plate gallery](experiments/figures/real_eval_gallery.png)

Per-image predictions (all 12, with regex-filtered strings and per-plate
char-accuracy where ground truth exists) are in
[`experiments/figures/real_eval_v1_v2.md`](experiments/figures/real_eval_v1_v2.md).

## Status

✅ Weekend 1 — renderer MVP
✅ Weekend 2 — synth-only training (ceiling: mAP@0.5:0.95 = 0.995)
✅ Weekend 3 — augmentation pipeline + augmented training (mAP@0.5:0.95 = 0.987)
✅ Weekend 4 — head-to-head on 12 real plates; augmentation lifts char-acc 4.4×
✅ Weekend 5a — public-source scrape: 2,418 unique real Thai plates from 9 Roboflow Universe projects
✅ Weekend 5b — VLM auto-labeler (Qwen2.5-VL-3B): 2,409/2,418 labeled, 1,285 high-confidence + regex-compliant
✅ Weekend 5c — hand-verified 50-plate gold set → **48% VLM exact plate accuracy, 88% char-level LCS**; all failures are consonant substitutions
✅ Weekend 5d — quality filter: drop 2 unreliable sources (86% / 67% skip rates) → **784-plate clean training corpus**
✅ Weekend 6 — confusion-aware synth + pseudo-label distillation: **negative result** — neither intervention improves on synth_v2's 0.346 char-acc, naive distillation regresses to 0.220
🚧 Weekend 7 — pick between (a) cleaner pseudo-labels via stage-1-driven bbox synthesis, (b) yolov8s backbone upgrade, or (c) replace stage-2 entirely with a small LoRA-tuned VLM

## Real-plate corpus

The 5-plate ground truth from `thai-plate-ocr` was too thin for the weekend-4
numbers to be statistically meaningful, so weekend 5 kicks off with a scrape
pass across public Thai-plate datasets on Roboflow Universe. **2,418 unique
images** after cross-source dedupe (pHash @ Hamming distance 5) from a
5,514-image raw pool. Full breakdown:
[`experiments/figures/scrape_summary.md`](experiments/figures/scrape_summary.md).

Images are gitignored; `src/thai_plate_synth/scrape/` holds the reproducible
scraper + deduper. Every surviving image has a provenance record
(workspace, project, version, license) in
`data/real_scrape/roboflow/provenance.jsonl`.

### VLM auto-labeling pass

Every scraped image is then read by **Qwen2.5-VL-3B-Instruct** (bf16 on the
3060 Ti, ~0.17 img/s, ~4 hours total). The VLM emits a structured
`REGISTRATION`/`PROVINCE`/`CONFIDENCE` triple per image, used as
**pseudo-ground-truth** for downstream student-model distillation / RL
reward. See [`experiments/figures/vlm_label_audit.md`](experiments/figures/vlm_label_audit.md)
for the full audit.

| metric | value |
|---|---:|
| labeled (non-error, non-empty) | 2,409 / 2,418 |
| high-confidence | 1,752 (73%) |
| regex-compliant overall | 1,480 (61%) |
| **usable corpus** (high ∩ regex-compliant) | **1,285** |

### Gold verification — the real accuracy number

A deterministic 50-plate sample (seed=42) from the 1,285 high-confidence
corpus was hand-verified via a [Streamlit UI](app/verify_labels.py). Every
verdict — correct / partial / wrong / unreadable-skip — is saved to
[`data/real_scrape/roboflow/gold_labels.jsonl`](data/real_scrape/roboflow/gold_labels.jsonl).

| verdict | count |
|---|---:|
| ✓ correct | 13 |
| ~ partial (consonant swap, digits right) | 11 |
| ✗ wrong | 3 |
| ⊘ skip (unreadable image) | 23 |

**VLM accuracy on the 27 rateable plates:**

| metric | value |
|---|---:|
| exact plate-level | 48.1% |
| with-partial-credit | 68.5% |
| char-level LCS | **88.0%** |
| digit accuracy | ~100% |
| consonant accuracy | ~67% — **the remaining bottleneck** |

**Every wrong/partial prediction is a Thai consonant substitution** on a
correct digit scaffold:

| VLM said | Ground truth | Confused pair |
|---|---|---|
| `6กพ3683` | `6กค3683` | พ↔ค |
| `ขด5011` | `ขฎ5011` | ด↔ฎ |
| `8กพ6487` | `8กฎ6487` | พ↔ฎ |
| `2กต8447` | `2กถ8447` | ต↔ถ |
| `1กพ571` | `1กผ571` | พ↔ผ |
| `มบ6224` | `ฆบ6224` | ม↔ฆ |

This is the *exact* failure mode hit by the original `thai-plate-ocr` YOLO
recognizer in the prior project.

## Weekend 6 — what didn't work (and why that's useful)

Two interventions tested, both ablated against `synth_v2`:

| Run | Delta | Config | Char-acc on gold-27 |
|---|---|---|---:|
| `synth_v1` | baseline | clean synth, no aug | 0.175 |
| `synth_v2` | — | + full augmentation | **0.346** |
| `synth_v3a` | +confusion | p_hard=0.3 oversample of ambiguous pairs | 0.331 |
| `synth_v3` | +confusion +pseudo | merge 282 real plates with synth_v2 bbox proposals | 0.220 |

**Finding 1 — confusion-aware sampling is neutral.** Oversampling the
visually-similar consonant pairs at p_hard=0.3 ran within noise of the
baseline. Naively "showing more of the hard characters" doesn't fix the
problem because the model's bottleneck isn't *exposure* to those glyphs
(it already sees each consonant ~100× per epoch) but its *capacity* to
resolve fine shape differences at `yolov8n` scale + `imgsz=480`.

**Finding 2 — naive pseudo-label distillation hurts.** Using
`synth_v2`'s bbox proposals on real plates as training targets
regressed char-acc by 0.126 points. synth_v2 is a weak detector on
real pixels, so its proposed bboxes are imprecise; training on those
noisy spatial targets taught synth_v3 to *under-detect* characters
(fragmented, shorter outputs). The class labels from the VLM were the
accurate part of the signal; the boxes were the liability.

These are both standard failure modes in self-distillation research,
confirmed here on a controlled ablation. The right next intervention
is one that produces cleaner bboxes (e.g., stage-1-detector crop + 
evenly-spaced chars) or avoids bbox supervision entirely (VLM SFT).

Full side-by-side per-plate predictions: [`experiments/figures/gold_eval.md`](experiments/figures/gold_eval.md).

### Source quality filter

The gold verification also exposed a per-source quality signal — one
Roboflow project (`thaich/thailand-license-plate-recognition-vx6tn`) had
an **86% unreadable rate** even when the VLM reported "high" confidence.
`src/thai_plate_synth/scrape/filter_corpus.py` drops it plus one other
unreliable source, producing **784 clean training plates**.

| filter stage | records |
|---|---:|
| raw VLM labels | 2,418 |
| high-confidence | 1,752 |
| high + regex-compliant | 1,285 |
| + source quality filter | **784** |

## License

- Code: MIT — see [LICENSE](LICENSE).
- Font: Sarun's ThangLuang, free for commercial use, **not redistributed**
  ([terms](https://www.f0nt.com/about/license/)). Download separately.
