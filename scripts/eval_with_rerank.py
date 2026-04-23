"""Evaluate synth_v3b with the confusion-aware reranker.

Runs the full chain: stage-1 detector → crop → stage-2 recognizer →
regex filter → trigram-LM reranker on consonant-substitution variants.

The reranker's `gap_threshold` controls how aggressively it overrides
YOLO's argmax: we only substitute a consonant if the LM-preferred
variant beats the YOLO-predicted variant by > `gap_threshold` in
log-probability. Threshold 0 is "pure LM rerank"; higher values fall
back to the YOLO prediction more often.

Outputs a markdown comparison over several thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from thai_plate_synth.rerank import (
    PLATE_PATTERN,
    TrigramLM,
    candidates,
    load_corpus_from_clean_labels,
)

# Reuse the eval helpers from scripts/eval_on_gold.py — the rerank step
# plugs in after the regex filter, the rest of the pipeline is identical.
import sys
sys.path.insert(0, str(Path(__file__).parent))
from eval_on_gold import CharBox, order_chars, _run_yolo, _string_from_boxes, _regex_filter  # noqa: E402

from thai_plate_synth.alphabet import CLASS_TO_GLYPH  # noqa: E402, F401


def _lcs_acc(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    sm = SequenceMatcher(None, pred, gt, autojunk=False)
    return sum(b.size for b in sm.get_matching_blocks()) / len(gt)


def rerank_with_gap(pred: str, lm: TrigramLM, gap_threshold: float) -> str:
    """Return the best LM-scored variant of pred, unless no variant beats
    pred's own log-prob by more than `gap_threshold`. When pred itself is
    among the candidates (always, since the confusion-set includes the
    identity), this never regresses below pred."""
    if not pred:
        return pred
    cands = [c for c in candidates(pred) if PLATE_PATTERN.match(c)]
    if not cands:
        return pred
    scored = [(c, lm.log_prob(c)) for c in cands]
    scored.sort(key=lambda t: t[1], reverse=True)
    best, best_score = scored[0]
    # Compare against pred's own score if pred is a valid candidate.
    pred_score = lm.log_prob(pred) if pred in dict(scored) else -float("inf")
    if best != pred and (best_score - pred_score) < gap_threshold:
        return pred
    return best


def eval_weights(
    weights_path: Path,
    images_dir: Path,
    gold_records: list[dict],
    detector_weights: Path,
    lm: TrigramLM,
    gap_thresholds: list[float],
    conf: float = 0.25,
    imgsz: int = 480,
) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    det = YOLO(str(detector_weights))

    rows_per_threshold: dict[float, list[dict]] = {t: [] for t in gap_thresholds}
    # Baseline (no rerank) tracked as threshold=None.
    rows_baseline: list[dict] = []

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
            x1, y1, x2, y2 = int(best.x1), int(best.y1), int(best.x2), int(best.y2)
            crop = img.crop((x1, y1, x2, y2)) if x2 > x1 and y2 > y1 else img
        else:
            crop = img

        boxes = _run_yolo(model, crop, conf, imgsz)
        raw = _string_from_boxes(boxes)
        pred = _regex_filter(raw) or raw
        gt = r["gold_registration"]

        rows_baseline.append({
            "image": r["image"], "gt": gt, "pred": pred, "acc": _lcs_acc(pred, gt),
        })
        for t in gap_thresholds:
            reranked = rerank_with_gap(pred, lm, t)
            rows_per_threshold[t].append({
                "image": r["image"], "gt": gt, "pred": reranked,
                "orig_pred": pred, "acc": _lcs_acc(reranked, gt),
            })

    def _summarize(rows: list[dict]) -> dict:
        n = len(rows)
        exact = sum(1 for r in rows if r["pred"] == r["gt"])
        total_acc = sum(r["acc"] for r in rows)
        return {
            "n": n, "exact_match": exact,
            "exact_rate": exact / n if n else 0.0,
            "char_acc": total_acc / n if n else 0.0,
        }

    return {
        "weights": str(weights_path),
        "baseline": {**_summarize(rows_baseline), "rows": rows_baseline},
        "rerank": {
            t: {**_summarize(rows_per_threshold[t]),
                "rows": rows_per_threshold[t]}
            for t in gap_thresholds
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path,
                    default=Path("experiments/runs/synth_v3b/weights/best.pt"))
    ap.add_argument("--detector", type=Path, default=Path(
        "/Users/zeemon/Development/thai-plate-ocr/runs/detect/artifacts/detector/train/weights/best.pt"
    ))
    ap.add_argument("--gold", type=Path, default=Path("data/real_scrape/roboflow/gold_labels.jsonl"))
    ap.add_argument("--images", type=Path, default=Path("data/real_scrape/roboflow/images"))
    ap.add_argument("--lm-corpus", type=Path, default=Path("data/real_scrape/roboflow/clean_labels.jsonl"))
    ap.add_argument("--gap-thresholds", type=float, nargs="+",
                    default=[0.0, 2.0, 5.0, 10.0])
    ap.add_argument("--out", type=Path, default=Path("experiments/figures/gold_eval_rerank.md"))
    args = ap.parse_args()

    # LM corpus — regex-filter registrations from the clean labels.
    corpus = load_corpus_from_clean_labels(args.lm_corpus)
    print(f"LM corpus: {len(corpus)} plate registrations from {args.lm_corpus}")

    # Exclude gold-27 registrations from LM to avoid test leakage.
    gold_records = []
    gold_regs_set: set[str] = set()
    for line in args.gold.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("verdict") != "skip" and r.get("gold_registration"):
                gold_records.append(r)
                gold_regs_set.add(r["gold_registration"])
    leak_count = sum(1 for s in corpus if s in gold_regs_set)
    corpus_clean = [s for s in corpus if s not in gold_regs_set]
    print(f"evaluating on {len(gold_records)} gold plates; "
          f"removed {leak_count} from LM corpus for leak-free eval")

    lm = TrigramLM(corpus_clean, alpha=0.5)

    res = eval_weights(args.weights, args.images, gold_records,
                       args.detector, lm, args.gap_thresholds)

    # Print summary
    base = res["baseline"]
    print(f"\nbaseline (no rerank): n={base['n']} exact={base['exact_match']}"
          f" ({base['exact_rate']:.1%}) char-acc={base['char_acc']:.3f}")
    for t in args.gap_thresholds:
        s = res["rerank"][t]
        print(f"gap={t:>5.1f}           : n={s['n']} exact={s['exact_match']}"
              f" ({s['exact_rate']:.1%}) char-acc={s['char_acc']:.3f}")

    # Markdown writeup
    md: list[str] = []
    md.append("# Gold-set eval — synth_v3b with confusion reranker")
    md.append("")
    md.append(f"Evaluated on {base['n']} hand-verified readable plates from the "
              f"Roboflow-scraped corpus.")
    md.append(f"Reranker: char-trigram LM trained on {len(corpus_clean)} real "
              f"plate registrations (gold-27 strings excluded for leak-free eval).")
    md.append("Gap threshold = minimum log-prob advantage the LM-preferred variant "
              "must have over the YOLO argmax before substituting.")
    md.append("")
    md.append("| config | n | exact | exact rate | char-acc |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| synth_v3b (no rerank) | {base['n']} | {base['exact_match']} | "
              f"{base['exact_rate']:.1%} | **{base['char_acc']:.3f}** |")
    for t in args.gap_thresholds:
        s = res["rerank"][t]
        md.append(f"| + rerank (gap={t:.1f}) | {s['n']} | {s['exact_match']} | "
                  f"{s['exact_rate']:.1%} | **{s['char_acc']:.3f}** |")
    md.append("")
    md.append("## Per-plate predictions")
    md.append("")
    head = ["gt", "synth_v3b"] + [f"gap={t:.1f}" for t in args.gap_thresholds]
    md.append("| " + " | ".join(head) + " |")
    md.append("|" + "|".join(["---"] * len(head)) + "|")
    for i in range(base["n"]):
        b = base["rows"][i]
        cells = [f"`{b['gt']}`", f"`{b['pred'] or '_'}` ({b['acc']:.2f})"]
        for t in args.gap_thresholds:
            r = res["rerank"][t]["rows"][i]
            changed = "*" if r["pred"] != r["orig_pred"] else ""
            cells.append(f"`{r['pred'] or '_'}`{changed} ({r['acc']:.2f})")
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("`*` denotes a plate where the reranker changed the YOLO argmax.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
