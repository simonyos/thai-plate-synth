# Real-plate scrape — public Roboflow Universe

Collected **5,514 raw images** across 9 public Thai-plate projects, then
deduplicated across sources with a 64-bit aHash at Hamming threshold 5 to
remove the substantial re-upload overlap between Universe projects.

**Final corpus: 2,418 unique real Thai-plate images**.

Images themselves live in `data/real_scrape/roboflow/images/` and are **not
committed** (gitignored). The scraper script and `provenance.jsonl` are the
reproducible artifacts; downstream consumers re-run the scrape locally.

## Per-source contribution (after dedupe)

| Source | Kept unique |
|---|---:|
| `thaich/thailand-license-plate-recognition-vx6tn` | 766 |
| `card-detector/thai-license-plate-character-detect` | 496 |
| `card-detector/thai-license-plate-wniws` | 437 |
| `nextra/thai-licence-plate-detect-b93xq` | 282 |
| `dataset-format-conversion-iidaz/thailand-license-plate-recognition` | 190 |
| `th-support-cytron-yvkeg/thai-license-plate-detector` | 125 |
| `naruesorn/thai-license-plate-j6y9l` | 108 |
| `th-support-cytron-yvkeg/province-on-thai-license-plate-detector` | 12 |
| `license-plate-q7bk1/thailand-license-plate` | 2 |
| **Total** | **2,418** |

## Skipped

Two projects had no published versions at scrape time:
- `thailland-plates/thailand-license-plates`
- `nutjulanan/province-on-thai-license-plate-detector-hntyj`

## Reproducing locally

```bash
export ROBOFLOW_API_KEY=<your-key>
uv run python -m thai_plate_synth.scrape.roboflow_scrape
uv run python -m thai_plate_synth.scrape.dedupe
```

Outputs: `data/real_scrape/roboflow/images/` + `provenance.jsonl` + `summary.json`.
