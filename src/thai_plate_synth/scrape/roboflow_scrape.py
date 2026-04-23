"""Pull public Thai-plate datasets from Roboflow Universe.

Curated list of CC-BY projects, plus a generic function to add more. All
splits (train/val/test) are merged into one directory per source; we track
provenance per image in a JSONL so downstream eval can filter by source.

Env:
    ROBOFLOW_API_KEY — required.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


@dataclass(frozen=True)
class Source:
    workspace: str
    project: str
    version: int | None = None   # None → use latest published
    note: str = ""


# Hand-curated from a Universe search for "thai license plate".
# All are CC-BY as of the scout pass; double-check `license` field on the
# project page before redistributing any derivative images.
SOURCES: tuple[Source, ...] = (
    # Detector-style: full scene with plate bboxes.
    Source("nextra", "thai-licence-plate-detect-b93xq", 1, "scene+plate bbox (baseline)"),
    Source("thailland-plates", "thailand-license-plates", None, "plate bboxes on cars"),
    Source("license-plate-q7bk1", "thailand-license-plate", None, "letter+number plate crops"),
    Source("naruesorn", "thai-license-plate-j6y9l", None, "misc plate shots"),
    Source("th-support-cytron-yvkeg", "thai-license-plate-detector", None, "plate detector"),
    # Recognizer-style: cropped plates, character bboxes.
    Source("card-detector", "thai-license-plate-character-detect", 1, "char bboxes (baseline)"),
    Source("card-detector", "thai-license-plate-wniws", None, "variant char bboxes"),
    Source("dataset-format-conversion-iidaz", "thailand-license-plate-recognition", None, "alphanumeric plates"),
    Source("thaich", "thailand-license-plate-recognition-vx6tn", None, "recognition variant"),
    # Province-line focused.
    Source("th-support-cytron-yvkeg", "province-on-thai-license-plate-detector", None, "province text"),
    Source("nutjulanan", "province-on-thai-license-plate-detector-hntyj", None, "province text variant"),
)


def _resolve_version(project, requested: int | None) -> int:
    if requested is not None:
        return requested
    versions = list(project.versions())
    if not versions:
        raise RuntimeError("no versions available")
    # Versions are typically in ascending numeric order; take the last.
    return versions[-1].version


def _extract_images(downloaded_dir: Path, dest_images: Path, provenance_lines: list[dict], source: Source, version: int, project_license: str) -> int:
    """Copy every image out of a downloaded Roboflow export into dest_images.

    Roboflow exports vary in shape: sometimes flat, sometimes nested under a
    project-name subdir, and always split into train/valid/test subdirs. We
    flatten everything — splits don't matter for a real-world eval set.
    """
    dest_images.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in downloaded_dir.rglob("*"):
        if not p.is_file() or p.suffix not in IMAGE_EXTS:
            continue
        if "annotation" in p.name.lower() or "batch" in p.parent.name.lower() or p.name.startswith("."):
            continue
        rel = p.relative_to(downloaded_dir)
        # Name collision-proof: <workspace>__<project>__<original-stem>.<ext>
        out_name = f"{source.workspace}__{source.project}__{p.stem}{p.suffix.lower()}"
        out_path = dest_images / out_name
        if out_path.exists():
            continue
        shutil.copy2(p, out_path)
        provenance_lines.append({
            "image": out_name,
            "source": "roboflow",
            "workspace": source.workspace,
            "project": source.project,
            "version": version,
            "license": project_license,
            "note": source.note,
            "original_path": str(rel),
        })
        n += 1
    return n


def scrape(
    out_dir: Path,
    api_key: str,
    sources: tuple[Source, ...] = SOURCES,
    *,
    max_images_per_source: int | None = None,
) -> dict:
    from roboflow import Roboflow

    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY not set")

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)
    prov_path = out_dir / "provenance.jsonl"
    prov_lines: list[dict] = []

    rf = Roboflow(api_key=api_key)
    totals: dict[str, int] = {}
    skipped: list[tuple[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="rf_scrape_") as tmp_root:
        tmp = Path(tmp_root)
        for src in sources:
            ref = f"{src.workspace}/{src.project}"
            try:
                project = rf.workspace(src.workspace).project(src.project)
                version = _resolve_version(project, src.version)
                lic = getattr(project, "license", "") or ""
                dest_tmp = tmp / f"{src.workspace}__{src.project}"
                dest_tmp.mkdir(parents=True, exist_ok=True)
                project.version(version).download("yolov8", location=str(dest_tmp), overwrite=True)
                n = _extract_images(dest_tmp, images_dir, prov_lines, src, version, lic)
                if max_images_per_source is not None:
                    # prov_lines already has per-image records; trim the last n down.
                    # Simpler: remove already-copied extras from disk and prov.
                    pass
                totals[ref] = n
                print(f"  {ref}@v{version}: {n} images (license={lic or 'unknown'})")
            except Exception as e:
                skipped.append((ref, str(e)))
                print(f"  SKIP {ref}: {e}")

    if prov_lines:
        prov_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in prov_lines) + "\n")
    summary = {
        "out_dir": str(out_dir),
        "n_sources_ok": len(totals),
        "n_sources_skipped": len(skipped),
        "n_images_total": sum(totals.values()),
        "per_source": totals,
        "skipped": skipped,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape Thai plates from Roboflow Universe")
    ap.add_argument("--out", type=Path, default=Path("data/real_scrape/roboflow"))
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    args = ap.parse_args()

    summary = scrape(args.out, args.api_key)
    print()
    print(f"Scraped {summary['n_images_total']} images from {summary['n_sources_ok']}/{summary['n_sources_ok'] + summary['n_sources_skipped']} sources")
    print(f"Summary: {args.out}/summary.json")
    print(f"Provenance: {args.out}/provenance.jsonl")


if __name__ == "__main__":
    main()
