# Model Setup

## DiffMS
Upstream: https://github.com/coleygroup/DiffMS  
Operating point: K1.

## FRIGID
Upstream: https://github.com/coleygroup/FRIGID  
Operating point: FRIGID-base@FM1.

- formula matches required: 1
- maximum attempts: 100
- batch size: 4
- temperature: 1.0
- randomness: 10.0
- fingerprint threshold: 0.187

The author-trained token-length NGBoost model is included under `artifacts/frigid_token_model/`.

## MS2Forge
Upstream: https://github.com/huaqiyao/MS2forge  
Operating point: K1 with 20 sampling timesteps.

## Third-party checkpoints

Third-party pretrained checkpoints are not redistributed. Exact filenames, sizes, and SHA-256 hashes are recorded in `provenance/checkpoint_hashes/checkpoint_sha256.csv`.
