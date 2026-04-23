"""Filter the VLM-labeled corpus down to a clean training subset.

Applies three stacked filters in order:

  1. Source filter — drop Roboflow projects whose gold-verification skip rate
     was prohibitively high (see experiments/figures/vlm_label_audit.md).
  2. Confidence filter — VLM confidence must be "high".
  3. Format filter — registration must match the canonical Thai plate regex.

Writes a clean_labels.jsonl (same schema as vlm_labels.jsonl) plus a summary.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PLATE_PATTERN = re.compile(r"^[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}$")

# Sources excluded because gold-verification showed they're unreadable by humans
# at the resolution/quality they were scraped at. Skip rates from a 50-plate
# hand-verified sample (data/real_scrape/roboflow/gold_labels.jsonl):
#   thaich/*            — 86% (18/21)   over-cropped / low-res plates
#   card-detector/*wniws — 67% (2/3)    same issue
EXCLUDE_PROJECTS: frozenset[str] = frozenset(
    {
        "thailand-license-plate-recognition-vx6tn",
        "thai-license-plate-wniws",
    }
)


def _normalize(reg: str) -> str:
    return re.sub(r"[^0-9ก-ฮ]", "", reg)


def filter_corpus(labels_path: Path, provenance_path: Path, out_path: Path) -> dict:
    # Load provenance as image→source map.
    prov: dict[str, dict] = {}
    for line in provenance_path.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
                prov[r["image"]] = r
            except Exception:
                pass

    kept: list[dict] = []
    drop_by_source: Counter[str] = Counter()
    drop_by_conf: Counter[str] = Counter()
    drop_by_regex: Counter[str] = Counter()
    per_source_kept: Counter[str] = Counter()

    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)

        # (1) Source filter
        p = prov.get(r["image"], {})
        project = p.get("project", "")
        if project in EXCLUDE_PROJECTS:
            drop_by_source[project] += 1
            continue

        # (2) Confidence filter
        conf = r.get("confidence", "")
        if conf != "high":
            drop_by_conf[conf or "unknown"] += 1
            continue

        # (3) Format filter
        reg = _normalize(r.get("registration", ""))
        if not PLATE_PATTERN.match(reg):
            drop_by_regex[str(len(reg))] += 1
            continue

        r["normalized_registration"] = reg
        kept.append(r)
        per_source_kept[project or "unknown"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n" if kept else "")

    return {
        "input_records": sum(1 for _ in labels_path.read_text().splitlines() if _.strip()),
        "kept": len(kept),
        "dropped_by_source": dict(drop_by_source),
        "dropped_by_confidence": dict(drop_by_conf),
        "dropped_by_regex": dict(drop_by_regex),
        "per_source_kept": dict(per_source_kept),
        "excluded_projects": sorted(EXCLUDE_PROJECTS),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/real_scrape/roboflow/vlm_labels.jsonl"))
    ap.add_argument("--provenance", type=Path, default=Path("data/real_scrape/roboflow/provenance.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/real_scrape/roboflow/clean_labels.jsonl"))
    args = ap.parse_args()
    res = filter_corpus(args.labels, args.provenance, args.out)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nClean corpus: {res['kept']} records → {args.out}")


if __name__ == "__main__":
    main()
