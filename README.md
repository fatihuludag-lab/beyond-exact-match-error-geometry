# Beyond Exact Match: Error Geometry in Molecular Generation from Tandem Mass Spectra

Reproducibility package for the article by **Fatih Uludağ**.

## Overview

This repository contains frozen development and independent confirmatory panel manifests, strict/harmonized predictions from DiffMS, FRIGID, and MS2Forge, exact MCES results, verified synthetic molecular corruptions, final statistical tables, environment snapshots, and provenance metadata.

The primary independent confirmatory population is **MSG-EG358-COMMON-SUPPORT-CONFIRM-v01**.

## Main confirmatory results

| Model | Task-valid | Same formula | Exact |
|---|---:|---:|---:|
| DiffMS-K1 | 158 / 358 | 38 / 358 | 0 / 358 |
| FRIGID-base@FM1 | 358 / 358 | 333 / 358 | 10 / 358 |
| MS2Forge-K1 | 226 / 358 | 108 / 358 | 39 / 358 |

On common valid but incorrect predictions, mean paired MCES differences were:
- FRIGID - DiffMS: -3.4904; 95% CI [-5.1506, -1.8365].
- MS2Forge - DiffMS: -6.1011; 95% CI [-7.6596, -4.6064].
- MS2Forge - FRIGID: -4.0467; 95% CI [-5.4066, -2.6813].

Across 333 matched parent molecules in the synthetic-real audit:
- median real-error MCES = 17.0;
- median synthetic-corruption MCES = 2.0;
- mean paired real-minus-synthetic MCES difference = 14.8168;
- 324 / 333 parents had more severe real errors.

## Repository layout

- `manifests/` frozen panel definitions and protocol locks.
- `data/normalized_predictions/` strict/harmonized model outputs.
- `data/synthetic_corruptions/` verified synthetic corruption corpus.
- `data/derived_analysis_tables/` final publication tables.
- `mces/` exact MCES mappings and outputs.
- `scripts/provenance/` execution scripts retained for provenance.
- `scripts/synthetic_corruptions/` synthetic corruption scripts.
- `provenance/` upstream Git state and checkpoint hashes.
- `environment/` Python and system snapshots.
- `checksums/` release inventory and SHA-256 checksums.

## Raw data and pretrained checkpoints

The full MassSpecGym spectra and third-party pretrained checkpoints are not redistributed.

Upstream sources:
- MassSpecGym: https://github.com/pluskal-lab/MassSpecGym
- DiffMS: https://github.com/coleygroup/DiffMS
- FRIGID: https://github.com/coleygroup/FRIGID
- MS2Forge: https://github.com/huaqiyao/MS2forge

Exact local Git commits, patches, environments, and checkpoint SHA-256 hashes used in this study are archived here.

## Verification

With pandas and pyarrow installed:

```powershell
python scripts\verify_release.py
```

Expected final line:

```text
ALL SCIENTIFIC RELEASE GATES PASSED.
```

## Repository

Source repository: https://github.com/fatihuludag-lab/beyond-exact-match-error-geometry

## Licensing

Author-written code is released under MIT. Author-generated derived data are intended for reuse under CC BY 4.0 where the author has the right to do so. Third-party materials retain their original licenses.

## Citation

GitHub repository: https://github.com/fatihuludag-lab/beyond-exact-match-error-geometry

Archived release: https://doi.org/10.5281/zenodo.23325150
