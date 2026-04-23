# VLM label audit

Total records: **2418**  
Labeled (non-error, non-empty): **2409**  
Regex-compliant: **1480** (61.4%)

## Confidence distribution
| level | count |
|---|---:|
| high | 1752 |
| medium | 1 |
| low | 664 |
| unknown | 1 |

## Registration length distribution
| chars | count |
|---:|---:|
| 1 | 2 |
| 2 | 24 |
| 3 | 62 |
| 4 | 307 |
| 5 | 277 |
| 6 | 625 |
| 7 | 1018 |
| 8 | 82 |
| 9 | 3 |
| 10 | 6 |
| 11 | 2 |
| 46 | 1 |

## Top predicted provinces
| province | count |
|---|---:|
| `กรุงเทพมหานคร` | 1681 |
| `-` | 165 |
| `ชลบุรี` | 10 |
| `นนทบุรี` | 10 |
| `ระยอง` | 4 |
| `ลพบุรี` | 3 |
| `ระนอง` | 3 |
| `บางเขน` | 3 |
| `พระนครศรีอยุธยา` | 2 |
| `ไม่ระบุ` | 2 |
| `ปราจีนบุรี` | 2 |
| `ขลบร.` | 2 |
| `ขลบุรี` | 2 |
| `ฉะเชิงเทรา` | 2 |
| `ขอนแก่น` | 2 |

## Per-source VLM error rate
| source | total | errors | error rate |
|---|---:|---:|---:|
| `card-detector/thai-license-plate-character-detect` | 496 | 0 | 0.0% |
| `card-detector/thai-license-plate-wniws` | 437 | 0 | 0.0% |
| `dataset-format-conversion-iidaz/thailand-license-plate-recognition` | 190 | 0 | 0.0% |
| `license-plate-q7bk1/thailand-license-plate` | 2 | 0 | 0.0% |
| `naruesorn/thai-license-plate-j6y9l` | 108 | 0 | 0.0% |
| `nextra/thai-licence-plate-detect-b93xq` | 282 | 0 | 0.0% |
| `th-support-cytron-yvkeg/province-on-thai-license-plate-detector` | 12 | 0 | 0.0% |
| `th-support-cytron-yvkeg/thai-license-plate-detector` | 125 | 0 | 0.0% |
| `thaich/thailand-license-plate-recognition-vx6tn` | 766 | 0 | 0.0% |

## Sample regex failures (first 10)
| image | raw registration |
|---|---|
| `card-detector__thai-license-plate-character-detect…` | `492` |
| `card-detector__thai-license-plate-character-detect…` | `292874` |
| `card-detector__thai-license-plate-character-detect…` | `25914400` |
| `card-detector__thai-license-plate-character-detect…` | `4ก65419` |
| `card-detector__thai-license-plate-character-detect…` | `4ก65419` |
| `card-detector__thai-license-plate-character-detect…` | `2ก54884` |
| `card-detector__thai-license-plate-character-detect…` | `4ก65928` |
| `card-detector__thai-license-plate-character-detect…` | `452278` |
| `card-detector__thai-license-plate-character-detect…` | `40282` |
| `card-detector__thai-license-plate-character-detect…` | `6999` |
