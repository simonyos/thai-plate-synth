"""Post-hoc re-parse of a vlm_labels.jsonl using the current _parse().

The VLM can output fields on collapsed lines (e.g. 'PROVINCE: CONFIDENCE: low'
on one line). Earlier runs captured this as a province value of 'CONFIDENCE:
low'; the updated parser in vlm_label.py strips these spillovers. This
script applies that fix to an existing labels file by re-running _parse on
each record's `raw_output`.

Usage:
    uv run python -m thai_plate_synth.scrape.reparse \
        --labels data/real_scrape/roboflow/vlm_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thai_plate_synth.scrape.vlm_label import _parse


def reparse(path: Path) -> dict:
    lines = path.read_text().splitlines()
    n_changed = 0
    out_lines: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        raw = r.get("raw_output", "")
        if not raw:
            out_lines.append(line)
            continue
        new = _parse(raw)
        if (
            new["registration"] != r.get("registration")
            or new["province"] != r.get("province")
            or new["confidence"] != r.get("confidence")
        ):
            n_changed += 1
        r.update(new)
        out_lines.append(json.dumps(r, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + "\n")
    return {"total": len(out_lines), "changed": n_changed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/real_scrape/roboflow/vlm_labels.jsonl"))
    args = ap.parse_args()
    res = reparse(args.labels)
    print(res)


if __name__ == "__main__":
    main()
