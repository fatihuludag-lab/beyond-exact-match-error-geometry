# Data Dictionary

## Normalized prediction tables

Files in `data/normalized_predictions/` contain one row per spectrum/model prediction.

Common fields:
- `spec`: spectrum identifier.
- `truth_smiles`: evaluation ground truth.
- `pred_smiles`: normalized prediction.
- `rdkit_valid`: fresh RDKit sanitization result where present.
- `n_components`: number of molecular components where present.
- `task_valid`: common strict task-valid criterion.
- `same_formula`: whether the prediction has the same molecular formula as the truth.
- `exact_connectivity`: connectivity-level exact recovery.
- `tanimoto`: Morgan fingerprint Tanimoto similarity.

## RQ2 exact MCES

`mces/rq2/EG358_CONFIRM_RQ2_MCES_MAP_MANIFEST_TRUTH.csv` maps each model-spectrum pair to the authoritative frozen-manifest truth and prediction.

`mces/rq2/EG358_CONFIRM_RQ2_MCES_OUT_MANIFEST_TRUTH.csv` is headerless with columns:
1. `pair_id`
2. `mces`
3. computation time in seconds
4. computation mode

All final confirmatory MCES calculations used exact mode.

## RQ4 synthetic corpus

`data/synthetic_corruptions/RQ4_CONFIRM_SYNTHETIC_PRIMARY_EG358.parquet` contains 3261 verified synthetic corruptions from 333 parent molecules.

Important fields include `sample_id`, `parent_spec`, `true_smiles`, `start_smiles`, `corruption_family`, `inverse_verified`, `inverse_truncated_same_family`, and `forward_truncated_family`.

## RQ4 parent-matched table

`data/derived_analysis_tables/RQ4_CONFIRM_FINAL_PARENT_MATCHED_EG358.csv` contains one row per matched parent and fields including:
`parent_spec`, `real_median_mces`, `synthetic_median_mces`, `real_median_tanimoto`, `synthetic_median_tanimoto`, `n_real_errors`, `n_synthetic`, `delta_mces_real_minus_synthetic`, and `delta_tanimoto_real_minus_synthetic`.
