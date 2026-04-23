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
✅ Weekend 7 — **clean-bbox distillation hits 0.721 char-acc** (2.1× the synth_v2 baseline); yolov8s backbone alone regresses
✅ Weekend 8 — post-processing hit its ceiling: confusion-LM reranker neutral (0.721 → 0.716), inference-time preprocessing sweep (imgsz + top-band recrop) neutral to negative. synth_v3b at 72.1% is the recognizer's real limit; further gains need retraining
🚧 Weekend 9 — Streamlit demo + HF Space + writeup

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

## Weekend 7 — the clean-bbox result

Two one-variable experiments against the synth_v2 baseline:

| Run | Change vs baseline | Gold-27 char-acc | Δ vs synth_v2 |
|---|---|---:|---:|
| `synth_v2` | — (baseline) | 0.346 | — |
| `synth_v4` | yolov8n → yolov8s backbone, same data | 0.286 | **−0.060** |
| `synth_v3b` | new pseudo-labels: detector-crop + evenly-spaced bboxes | **0.721** | **+0.375** |

Weekend 6's conclusion — *"the class labels from the VLM are the accurate part
of the signal; the bboxes are the liability"* — motivated the synth_v3b
pipeline rewrite:

1. Run the stage-1 [`thai-plate-ocr`](https://github.com/simonyos/thai-plate-ocr)
   plate detector to crop each real image to the plate region.
2. Discard the synth_v2 character-box proposals entirely.
3. For each crop, split the width into `len(gt_string)` equal slices; assign
   each slice the character class from the VLM string, with a generous
   vertical band chosen from the crop aspect ratio (2-line vs 1-line plate).

The resulting bboxes are spatially imprecise — Thai digits are narrower than
consonants, so the slices don't align tightly — but the class labels are
correct and the horizontal ordering is correct, which is what the classifier
head actually needs. And crucially, *every* record survives: 784/784 kept
(vs 282/784 with the synth_v2-proposal alignment) because we're no longer
bottlenecked on a weak model's detection count matching the VLM string
length.

**Backbone upgrade underperformed** — yolov8s (11M params vs yolov8n's 3M)
trained to higher synth-set mAP@0.5:0.95 (0.993 vs 0.987) but lost ground
on real plates. Two exact matches (`8กฎ6487`, `6กม3928` — the cleanest
straight-on shots) show the extra capacity does land on easy cases, but
it overfit the synthetic distribution enough to hurt broad generalisation.
Confirms weekend-6's diagnosis: the bottleneck isn't parameter count, it's
real-world pixel exposure.

**Key synth_v3b gains** from the per-plate view ([full table](experiments/figures/gold_eval.md)):

| gt | synth_v2 | synth_v3b |
|---|---|---|
| `6กค3683` | `ฮ` (0.00) | `6ก3683` (0.86) |
| `ทษ4346` | `พพพฝ` (0.00) | `43466` (0.67) |
| `ธษ456` | `1` (0.00) | `รข4456` (0.60) |
| `2กณ6969` | `4` (0.00) | `2กต6969` (0.86) |
| `4กน8869` | `ฐ` (0.00) | `4ก8869` (0.86) |
| `1กฌ1616` | `4` (0.00) | `1กพ1616` (0.86) |
| `6กค3683` | `ภคก3683` (0.71) | `66กค3683` (**1.00**) |

synth_v3b doesn't break the zero-exact-match barrier (one near-miss hits
LCS=1.0 but with a duplicated leading digit), and the remaining errors are
still consonant substitutions from the weekend-5 consonant-confusion
catalogue (ค↔พ, ฎ↔ด, ต↔ถ). But it *dramatically* narrows those errors
from "plate is unreadable" to "plate has one wrong consonant among 6–7
correct characters" — the shape of the remaining gap is now a small,
well-defined language-model or lookup-table problem rather than a vision
problem.

Full side-by-side per-plate predictions: [`experiments/figures/gold_eval.md`](experiments/figures/gold_eval.md).

## Weekend 8 — where post-processing tapped out

Two cheap non-training interventions, both failed to budge synth_v3b's
0.721 ceiling:

| Intervention | Best variant | Char-acc | Δ |
|---|---|---:|---:|
| synth_v3b (baseline) | — | 0.721 | — |
| Trigram-LM consonant reranker | gap-threshold ∈ {0, 2, 5} | 0.716 | **−0.005** |
| Trigram-LM reranker (bigger corpus) | 1,480 VLM labels | 0.710 | **−0.011** |
| Inference imgsz sweep | imgsz=640 no-recrop | 0.706 | −0.015 |
| Inference imgsz sweep | imgsz=960 | 0.321 | −0.400 |
| Top-band re-crop (2-line plates) | any imgsz | 0.671–0.678 | −0.050 |

**Why the reranker failed:** It was designed against the weekend-5 audit's
"errors are consonant substitutions" finding — which held for the *VLM*
but not for synth_v3b. Per-plate inspection showed synth_v3b's errors
are a *mix* of:
- missing/extra characters (length mismatches) — reranker can't fix
- wrong digits — outside the confusion catalogue
- consonants outside the confusion catalogue (ณ, ล, ฉ, ฐ, …)
- only ~3 of 27 plates had errors the reranker could theoretically fix,
  and on one of those it actively overrode a *correct* YOLO prediction
  toward a more-frequent consonant (`66กค3683` → `66กพ3683`) because
  the LM's frequency prior beat YOLO's local evidence.

**Why preprocessing failed:** synth_v3b was trained at `imgsz=480`; larger
test imgsz trades receptive-field alignment for resolution, and the
tradeoff went the wrong way. Top-band re-crop (matching the training
label distribution, which only spanned the top band on 2-line plates)
also hurt — the recognizer apparently uses *some* global context from
the full crop even when bboxes never landed in the bottom half.

Both are reported with full per-plate tables in
[`gold_eval_rerank.md`](experiments/figures/gold_eval_rerank.md) and
[`gold_eval_preproc.md`](experiments/figures/gold_eval_preproc.md). The
code ships (`src/thai_plate_synth/rerank.py`,
`scripts/eval_preproc_sweep.py`) — both are useful templates and both
are honest negative signals.

**Read:** synth_v3b at 72.1% char-acc is the real ceiling for this
recognizer without retraining. The remaining 27.9% gap is a retraining-
or architecture-level problem, not a post-processing one.

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
