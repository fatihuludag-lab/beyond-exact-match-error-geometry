# Reproducibility

## Level 1 — verify archived results

Run:

```powershell
python scripts\verify_release.py
```

This verifies frozen panel sizes, strict model metrics, final RQ2 effects, final RQ4 parent-matched results, and exact-mode MCES counts.

## Level 2 — recompute statistical summaries

Use the archived normalized predictions and exact MCES outputs under `data/` and `mces/`.

Final confirmatory truth-dependent analyses use the frozen EG358 manifest as the authoritative structural reference.

## Level 3 — reproduce model inference

Obtain source code and pretrained checkpoints from the upstream projects listed in `MODEL_SETUP.md`. Exact Git state, local patches, environments, and checkpoint hashes are archived under `provenance/` and `environment/`.

### DiffMS execution note

The initial Windows confirmatory execution with a multiprocessing DataLoader stalled after the first evaluation batch. The complete canonical run used `num_workers=0` and `pin_memory=false`. DiffMS sampling differed across executions; only the complete canonical run is used in confirmatory analyses. See `manifests/DIFFMS_EG358_CONFIRM_RUN_PROVENANCE.json`.
