"""Audit VLM-generated labels for quality signals.

The VLM is a pseudo-labeler — we need independent checks on whether its
output is usable before feeding it to downstream training. This script
reports:

  - Regex-compliance rate on the registration line
  - Confidence distribution (high / medium / low / error)
  - Registration length distribution
  - Top-N province predictions
  - Cross-source error rate (which scraped source gave us the most VLM errors)

Writes a markdown report + prints a one-line pass/fail verdict.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PLATE_PATTERN = re.compile(r"^[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}$")


def _normalize(reg: str) -> str:
    """Strip whitespace and keep only consonant/digit glyphs for regex match."""
    return re.sub(r"[^0-9ก-ฮ]", "", reg)


def audit(labels_path: Path, provenance_path: Path | None = None) -> dict:
    prov = {}
    if provenance_path and provenance_path.is_file():
        for line in provenance_path.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    prov[r["image"]] = r
                except Exception:
                    pass

    conf_counts: Counter[str] = Counter()
    len_counts: Counter[int] = Counter()
    provinces: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    source_errors: Counter[str] = Counter()
    regex_pass = 0
    regex_fail_examples: list[tuple[str, str]] = []
    total = 0
    empty_reg = 0

    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            total += 1
            conf = r.get("confidence", "unknown")
            conf_counts[conf] += 1

            src_key = "unknown"
            if r["image"] in prov:
                pr = prov[r["image"]]
                src_key = f"{pr.get('workspace','?')}/{pr.get('project','?')}"
            sources[src_key] += 1

            if conf == "error":
                source_errors[src_key] += 1
                continue

            reg = _normalize(r.get("registration", ""))
            if not reg:
                empty_reg += 1
                continue
            len_counts[len(reg)] += 1
            prov_name = r.get("province", "").strip()
            if prov_name:
                provinces[prov_name] += 1

            if PLATE_PATTERN.match(reg):
                regex_pass += 1
            else:
                if len(regex_fail_examples) < 10:
                    regex_fail_examples.append((r["image"], reg))

    labeled = total - conf_counts.get("error", 0) - empty_reg
    regex_rate = regex_pass / labeled if labeled else 0.0

    return {
        "total_records": total,
        "labeled": labeled,
        "empty_registration": empty_reg,
        "confidence": dict(conf_counts),
        "registration_length": dict(sorted(len_counts.items())),
        "top_provinces": provinces.most_common(15),
        "regex_pass": regex_pass,
        "regex_pass_rate": regex_rate,
        "regex_fail_examples": regex_fail_examples,
        "per_source": dict(sources),
        "per_source_errors": dict(source_errors),
    }


def write_report(result: dict, out: Path) -> None:
    lines = ["# VLM label audit", ""]
    lines.append(f"Total records: **{result['total_records']}**  ")
    lines.append(f"Labeled (non-error, non-empty): **{result['labeled']}**  ")
    lines.append(f"Regex-compliant: **{result['regex_pass']}** ({result['regex_pass_rate']:.1%})")
    lines.append("")

    lines.append("## Confidence distribution")
    lines.append("| level | count |")
    lines.append("|---|---:|")
    for k in ("high", "medium", "low", "unknown", "error"):
        if k in result["confidence"]:
            lines.append(f"| {k} | {result['confidence'][k]} |")
    lines.append("")

    lines.append("## Registration length distribution")
    lines.append("| chars | count |")
    lines.append("|---:|---:|")
    for k, v in result["registration_length"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Top predicted provinces")
    lines.append("| province | count |")
    lines.append("|---|---:|")
    for name, n in result["top_provinces"]:
        lines.append(f"| `{name}` | {n} |")
    lines.append("")

    lines.append("## Per-source VLM error rate")
    lines.append("| source | total | errors | error rate |")
    lines.append("|---|---:|---:|---:|")
    for src, n in result["per_source"].items():
        err = result["per_source_errors"].get(src, 0)
        rate = err / n if n else 0
        lines.append(f"| `{src}` | {n} | {err} | {rate:.1%} |")
    lines.append("")

    if result["regex_fail_examples"]:
        lines.append("## Sample regex failures (first 10)")
        lines.append("| image | raw registration |")
        lines.append("|---|---|")
        for img, reg in result["regex_fail_examples"]:
            lines.append(f"| `{img[:50]}…` | `{reg}` |")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/real_scrape/roboflow/vlm_labels.jsonl"))
    ap.add_argument("--provenance", type=Path, default=Path("data/real_scrape/roboflow/provenance.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("experiments/figures/vlm_label_audit.md"))
    args = ap.parse_args()
    res = audit(args.labels, args.provenance)
    write_report(res, args.out)
    print(f"labeled: {res['labeled']} / {res['total_records']}")
    print(f"regex-pass rate: {res['regex_pass_rate']:.1%}")
    print(f"confidence: {res['confidence']}")
    print(f"top 3 provinces: {res['top_provinces'][:3]}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
