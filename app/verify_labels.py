"""Streamlit UI to hand-verify VLM-generated plate labels.

The 50-plate sample (seed=42) becomes the gold test set. Every future
experiment is measured against it — so labels must be human-confirmed.

Run:
    uv run streamlit run app/verify_labels.py

Output:
    data/real_scrape/roboflow/gold_labels.jsonl — one record per verdict.
"""

from __future__ import annotations

import datetime as dt
import json
import random
import re
from pathlib import Path

import streamlit as st
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
LABELS_PATH = REPO / "data/real_scrape/roboflow/vlm_labels.jsonl"
IMAGES_DIR = REPO / "data/real_scrape/roboflow/images"
GOLD_PATH = REPO / "data/real_scrape/roboflow/gold_labels.jsonl"

PLATE_PATTERN = re.compile(r"^[0-9]{0,2}[ก-ฮ]{1,3}[0-9]{1,4}$")
N_SAMPLE = 50
SEED = 42


def _normalize(reg: str) -> str:
    return re.sub(r"[^0-9ก-ฮ]", "", reg)


@st.cache_data
def load_sample() -> list[dict]:
    """Deterministic random sample of N_SAMPLE from high-conf + regex-ok records."""
    records: list[dict] = []
    with LABELS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("confidence") != "high":
                continue
            reg = _normalize(r.get("registration", ""))
            if not PLATE_PATTERN.match(reg):
                continue
            r["normalized_registration"] = reg
            records.append(r)
    rng = random.Random(SEED)
    rng.shuffle(records)
    return records[:N_SAMPLE]


def _load_gold() -> dict[str, dict]:
    """Image name → verdict record (for resumption)."""
    out: dict[str, dict] = {}
    if GOLD_PATH.is_file():
        for line in GOLD_PATH.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    out[r["image"]] = r
                except Exception:
                    continue
    return out


def _append_verdict(record: dict) -> None:
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _rewrite_gold(records: list[dict]) -> None:
    """Full rewrite — used when correcting a prior verdict."""
    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLD_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n" if records else ""
    )


def main() -> None:
    st.set_page_config(page_title="Verify plate labels", layout="wide")
    st.title("Verify VLM plate labels")

    sample = load_sample()
    gold = _load_gold()

    # Find next unverified index (or requested index)
    if "idx" not in st.session_state:
        # Default to first unverified
        i = 0
        for j, r in enumerate(sample):
            if r["image"] not in gold:
                i = j
                break
        st.session_state.idx = i

    total = len(sample)
    done = sum(1 for r in sample if r["image"] in gold)
    i = st.session_state.idx
    rec = sample[i]
    img_path = IMAGES_DIR / rec["image"]

    # Header / progress
    cols = st.columns([2, 1, 1, 1])
    cols[0].progress(done / total, text=f"{done} / {total} verified")
    if cols[1].button("⬅ Prev", disabled=i == 0, use_container_width=True):
        st.session_state.idx = max(0, i - 1)
        st.rerun()
    if cols[2].button("Next ➡", disabled=i >= total - 1, use_container_width=True):
        st.session_state.idx = min(total - 1, i + 1)
        st.rerun()
    if cols[3].button("Jump to next unverified", use_container_width=True):
        for j, r in enumerate(sample):
            if r["image"] not in gold:
                st.session_state.idx = j
                st.rerun()

    st.markdown(f"**Plate {i + 1} / {total}** — `{rec['image'][:70]}{'…' if len(rec['image']) > 70 else ''}`")

    # Image + label side by side
    img_col, meta_col = st.columns([2, 1])
    with img_col:
        if img_path.is_file():
            img = Image.open(img_path).convert("RGB")
            st.image(img, use_container_width=True)
        else:
            st.error(f"Image not found: {img_path}")

    with meta_col:
        st.markdown("### VLM prediction")
        st.markdown(f"**Registration**: `{rec['normalized_registration']}`")
        if rec.get("province"):
            st.markdown(f"**Province**: `{rec['province']}`")
        st.markdown(f"**Confidence**: `{rec.get('confidence', '?')}`")

        st.markdown("### Source")
        st.caption(rec.get("raw_output", "")[:240])

        existing = gold.get(rec["image"])
        if existing:
            st.success(f"Already verified: **{existing['verdict']}**  → `{existing.get('gold_registration', '')}`")

    # Verdict UI
    st.markdown("### Verdict")
    verdict_cols = st.columns(4)
    correct_clicked = verdict_cols[0].button("✓ Correct", type="primary", use_container_width=True)
    partial_clicked = verdict_cols[1].button("~ Partial", use_container_width=True)
    wrong_clicked = verdict_cols[2].button("✗ Wrong", use_container_width=True)
    skip_clicked = verdict_cols[3].button(
        "⊘ Skip (unreadable)",
        use_container_width=True,
        help="Plate is too blurred / occluded / cropped to verify — excluded from accuracy tally",
    )

    # Correction field
    default_correction = (existing.get("gold_registration") if existing else rec["normalized_registration"]) or ""
    correction = st.text_input(
        "Correct registration (only needed if Wrong or Partial)",
        value=default_correction,
        key=f"correct_{rec['image']}",
    )

    clicked_verdict = None
    if correct_clicked:
        clicked_verdict = "correct"
    elif partial_clicked:
        clicked_verdict = "partial"
    elif wrong_clicked:
        clicked_verdict = "wrong"
    elif skip_clicked:
        clicked_verdict = "skip"

    if clicked_verdict:
        gold_reg = (
            rec["normalized_registration"]
            if clicked_verdict == "correct"
            else "" if clicked_verdict == "skip"
            else correction.strip()
        )
        new_record = {
            "image": rec["image"],
            "vlm_registration": rec["normalized_registration"],
            "vlm_province": rec.get("province", ""),
            "vlm_confidence": rec.get("confidence", ""),
            "verdict": clicked_verdict,
            "gold_registration": gold_reg,
            "verified_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }

        if existing:
            # Overwrite: rebuild file.
            merged = list(gold.values())
            merged = [r if r["image"] != rec["image"] else new_record for r in merged]
            _rewrite_gold(merged)
        else:
            _append_verdict(new_record)

        # Advance to next unverified
        gold = _load_gold()  # refresh
        for j in range(i + 1, total):
            if sample[j]["image"] not in gold:
                st.session_state.idx = j
                st.rerun()
                return
        # None left: stay on the last
        st.success("All 50 verified! Run the audit summary cell below.")
        st.rerun()

    # Footer — verdict tallies
    if gold:
        tallies = {"correct": 0, "partial": 0, "wrong": 0, "skip": 0}
        for r in gold.values():
            v = r.get("verdict", "")
            tallies[v] = tallies.get(v, 0) + 1
        st.markdown("---")
        st.markdown(
            f"Running tally: **{tallies.get('correct', 0)}** correct / "
            f"**{tallies.get('partial', 0)}** partial / "
            f"**{tallies.get('wrong', 0)}** wrong / "
            f"**{tallies.get('skip', 0)}** skipped (unreadable)"
        )
        if done == total:
            rated = total - tallies.get("skip", 0)
            if rated > 0:
                acc = tallies.get("correct", 0) / rated
                partial_credit = (tallies.get("correct", 0) + 0.5 * tallies.get("partial", 0)) / rated
                st.markdown(
                    f"**VLM plate-level accuracy** (n={rated}, skips excluded): "
                    f"exact = {acc:.1%}, with-partial-credit = {partial_credit:.1%}"
                )


if __name__ == "__main__":
    main()
