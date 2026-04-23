# Real-plate eval — synth_v1 (clean) vs synth_v2 (augmented)

Reproducibly sampled 12 images from the detector validation split (seed=7).
`v1`/`v2` columns are the longest Thai-plate-format regex match on the
raw recognizer output (i.e., the format-prior post-processing step — 
item #3 in the prior project's Next Steps list).
Character accuracy is LCS / |gt|, averaged across the 5 hand-labeled plates.

**synth_v1 (no aug):** char-acc = **0.15**, exact match = 0/5
**synth_v2 (augmented):** char-acc = **0.66**, exact match = 0/5

| idx | file | det conf | v1 pred | v2 pred | ground truth | v1 acc | v2 acc |
|---:|---|---:|---|---|---|---:|---:|
| 0 | `462742216_122211766808225440_414…` | 0.77 | `_` | `ป04` | `_` | — | — |
| 1 | `460437911_1070906627713039_60699…` | 0.53 | `ฆ1` | `ก121` | `_` | — | — |
| 2 | `463745514_449162348199564_282429…` | 0.94 | `กพ1` | `6กฟ7414` | `6กพ7414` | 0.43 | 0.86 |
| 3 | `463955985_2592010531189433_50342…` | 0.86 | `_` | `ฮกฝ031` | `8กย403` | 0.00 | 0.50 |
| 4 | `15_jpg.rf.ed60c99076d8b04754c21f…` | 0.76 | `_` | `ฮฉ6231` | `_` | — | — |
| 5 | `17_jpg.rf.8a9c2f67989d20ea33b5e9…` | 0.56 | `_` | `_` | `_` | — | — |
| 6 | `46_jpg.rf.92a6962fca23ea9f5aa115…` | 0.38 | `_` | `_` | `_` | — | — |
| 7 | `463872921_8723046997751287_10596…` | 0.80 | `_` | `5ก0` | `5กง9640` | 0.00 | 0.43 |
| 8 | `34_jpg.rf.7d28f0f572f5960aafbd3d…` | 0.83 | `_` | `83ธ2455` | `_` | — | — |
| 9 | `463740486_122142940442386340_320…` | 0.93 | `_` | `ฉฎ938` | `ฆฎ8938` | 0.00 | 0.67 |
| 10 | `463885863_10226968032415930_1797…` | 0.70 | `_` | `ฟ104` | `_` | — | — |
| 11 | `50_jpg.rf.18eb6a68277191fa6705e3…` | 0.84 | `ว00` | `ว4099` | `วฐ4099` | 0.33 | 0.83 |
