from pathlib import Path
import csv
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(r"C:\research\ms_error_geometry")

MANIFEST = ROOT / "manifests" / "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv"

RQ2_MAP = ROOT / "results" / "mces" / "EG358_CONFIRM_RQ2_MCES_MAP.csv"
RQ2_OUT = ROOT / "results" / "mces" / "EG358_CONFIRM_RQ2_MCES_OUT.csv"

REAL = ROOT / "results" / "RQ4_CONFIRM_REAL_THREE_MODELS_EG358.csv"
SYN  = ROOT / "results" / "RQ4_CONFIRM_SYNTHETIC_PRIMARY_EG358.parquet"

RQ2_FIX_MAP = ROOT / "results" / "mces" / "RQ2_CONFIRM_TRUTH_FIX_MAP.csv"
RQ2_FIX_INPUT = ROOT / "results" / "mces" / "RQ2_CONFIRM_TRUTH_FIX_INPUT.csv"

REAL_REUSE = ROOT / "results" / "mces" / "RQ4_CONFIRM_REAL_REUSED_FROM_RQ2.csv"
REAL_MISSING_MAP = ROOT / "results" / "mces" / "RQ4_CONFIRM_REAL_MISSING_MAP.csv"
REAL_MISSING_INPUT = ROOT / "results" / "mces" / "RQ4_CONFIRM_REAL_MISSING_INPUT.csv"

SYN_MAP = ROOT / "results" / "mces" / "RQ4_CONFIRM_SYNTHETIC_MAP.csv"
SYN_INPUT = ROOT / "results" / "mces" / "RQ4_CONFIRM_SYNTHETIC_INPUT.csv"

MODEL_ALIAS = {
    "DiffMS-K1": "DiffMS",
    "FRIGID-base@FM1": "FRIGID",
    "MS2Forge-K1": "MS2Forge",
}

FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def mol(s):
    m = Chem.MolFromSmiles(str(s), sanitize=False)
    if m is None:
        raise RuntimeError(f"Invalid SMILES: {s}")
    Chem.SanitizeMol(m)
    return m


def canonical(s):
    m = mol(s)
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(
        m,
        canonical=True,
        isomericSmiles=False
    )


def conn_key(s):
    m = mol(s)
    Chem.RemoveStereochemistry(m)
    k = Chem.MolToInchiKey(m)
    return k[:14] if k else None


def tani(a, b):
    ma = mol(a)
    mb = mol(b)
    return float(
        DataStructs.TanimotoSimilarity(
            FPGEN.GetFingerprint(ma),
            FPGEN.GetFingerprint(mb)
        )
    )


# ============================================================
# Authoritative frozen truth
# ============================================================

manifest = pd.read_csv(MANIFEST)

truth_col = next(
    c for c in ["canonical_smiles", "smiles", "truth_smiles"]
    if c in manifest.columns
)

manifest["spec"] = manifest["spec"].astype(str)

manifest["manifest_truth"] = (
    manifest[truth_col].apply(canonical)
)

manifest["manifest_key"] = (
    manifest["manifest_truth"].apply(conn_key)
)

manifest_lookup = (
    manifest.set_index("spec")
)


# ============================================================
# Existing RQ2 exact MCES
# ============================================================

rq2_map = pd.read_csv(RQ2_MAP)

rq2_out = pd.read_csv(
    RQ2_OUT,
    header=None,
    names=["pair_id", "mces", "time_sec", "mode"]
)

rq2 = rq2_map.merge(
    rq2_out,
    on="pair_id",
    validate="one_to_one"
)

assert len(rq2) == 585
assert set(rq2["mode"]) == {1}

rq2["manifest_truth"] = (
    rq2["spec"]
    .astype(str)
    .map(manifest_lookup["manifest_truth"])
)

rq2["rq2_truth_canonical"] = (
    rq2["truth_smiles"].apply(canonical)
)

rq2["rq2_truth_key"] = (
    rq2["truth_smiles"].apply(conn_key)
)

rq2["manifest_truth_key"] = (
    rq2["manifest_truth"].apply(conn_key)
)


connectivity_bad = rq2[
    rq2["rq2_truth_key"]
    != rq2["manifest_truth_key"]
]

print("=" * 100)
print("RQ2 TRUTH REFERENCE AUDIT")
print("=" * 100)

print(
    "RQ2 model-spec pairs =",
    len(rq2)
)

print(
    "Connectivity mismatches =",
    len(connectivity_bad)
)

if len(connectivity_bad):
    raise RuntimeError(
        "RQ2 truth connectivity differs from frozen manifest."
    )


rq2_fix = rq2[
    rq2["rq2_truth_canonical"]
    != rq2["manifest_truth"]
].copy()

print(
    "Representation-mismatched model-spec pairs =",
    len(rq2_fix)
)

print(
    "Unique affected spectra =",
    rq2_fix["spec"].nunique()
)


# These MCES values must be recomputed against frozen truth.
rq2_fix = rq2_fix.reset_index(drop=True)
rq2_fix["fix_pair_id"] = range(len(rq2_fix))

rq2_fix.to_csv(
    RQ2_FIX_MAP,
    index=False
)

with RQ2_FIX_INPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    for r in rq2_fix.itertuples(index=False):
        writer.writerow([
            r.fix_pair_id,
            r.manifest_truth,
            canonical(r.pred_smiles),
        ])


# ============================================================
# RQ4 real reuse audit
# ============================================================

real = pd.read_csv(REAL)

assert len(real) == 430
assert real["spec"].nunique() == 333

real["rq2_model"] = (
    real["model"].map(MODEL_ALIAS)
)

rq2_lookup = rq2.set_index(
    ["spec", "model"]
)

reused = []
missing = []

missing_truth_repr = 0
missing_pred_repr = 0
missing_no_rq2 = 0


for r in real.itertuples(index=False):

    key = (
        str(r.spec),
        str(r.rq2_model)
    )

    if key not in rq2_lookup.index:
        missing_no_rq2 += 1

        missing.append({
            "model": r.model,
            "spec": r.spec,
            "parent_spec": r.spec,
            "truth_smiles": canonical(r.truth_smiles),
            "pred_smiles": canonical(r.pred_smiles),
            "tanimoto": float(r.tanimoto),
            "reason": "not_in_RQ2"
        })

        continue


    old = rq2_lookup.loc[key]

    truth_now = canonical(r.truth_smiles)
    truth_old = canonical(old["truth_smiles"])

    pred_now = canonical(r.pred_smiles)
    pred_old = canonical(old["pred_smiles"])


    # Connectivity must agree.
    if conn_key(truth_now) != conn_key(truth_old):
        raise RuntimeError(
            f"Truth connectivity mismatch for {key}"
        )

    if conn_key(pred_now) != conn_key(pred_old):
        raise RuntimeError(
            f"Prediction connectivity mismatch for {key}"
        )


    # Reuse MCES only for identical graph representations.
    if truth_now != truth_old:
        missing_truth_repr += 1

        missing.append({
            "model": r.model,
            "spec": r.spec,
            "parent_spec": r.spec,
            "truth_smiles": truth_now,
            "pred_smiles": pred_now,
            "tanimoto": float(r.tanimoto),
            "reason": "truth_representation_changed"
        })

        continue


    if pred_now != pred_old:
        missing_pred_repr += 1

        missing.append({
            "model": r.model,
            "spec": r.spec,
            "parent_spec": r.spec,
            "truth_smiles": truth_now,
            "pred_smiles": pred_now,
            "tanimoto": float(r.tanimoto),
            "reason": "prediction_representation_changed"
        })

        continue


    reused.append({
        "model": r.model,
        "spec": r.spec,
        "parent_spec": r.spec,
        "truth_smiles": truth_now,
        "pred_smiles": pred_now,
        "tanimoto": float(r.tanimoto),
        "mces": float(old["mces"]),
        "mces_source": "RQ2_exact_reuse"
    })


reused = pd.DataFrame(reused)
missing = pd.DataFrame(missing)


print()
print("=" * 100)
print("RQ4 REAL MCES REUSE AUDIT")
print("=" * 100)

print("REAL TOTAL =", len(real))
print("REUSED EXACT MCES FROM RQ2 =", len(reused))
print("NEW REAL MCES REQUIRED =", len(missing))

print(
    "  not in RQ2 =",
    missing_no_rq2
)

print(
    "  truth representation changed =",
    missing_truth_repr
)

print(
    "  prediction representation changed =",
    missing_pred_repr
)

assert len(reused) + len(missing) == 430


reused.to_csv(
    REAL_REUSE,
    index=False
)


if len(missing):

    missing = missing.reset_index(drop=True)
    missing["pair_id"] = range(len(missing))

    missing.to_csv(
        REAL_MISSING_MAP,
        index=False
    )

    with REAL_MISSING_INPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        for r in missing.itertuples(index=False):
            writer.writerow([
                r.pair_id,
                r.truth_smiles,
                r.pred_smiles,
            ])


# ============================================================
# RQ4 synthetic
# ============================================================

syn = pd.read_parquet(SYN)

assert len(syn) == 3261
assert syn["parent_spec"].nunique() == 333

candidate_col = next(
    c
    for c in [
        "start_smiles",
        "corrupted_smiles",
        "candidate_smiles",
        "pred_smiles",
    ]
    if c in syn.columns
)

syn_rows = []

for pair_id, r in enumerate(
    syn.itertuples(index=False)
):

    truth = canonical(r.true_smiles)

    pred = canonical(
        getattr(r, candidate_col)
    )

    syn_rows.append({
        "pair_id": pair_id,
        "sample_id": r.sample_id,
        "parent_spec": r.parent_spec,
        "corruption_family": r.corruption_family,
        "truth_smiles": truth,
        "pred_smiles": pred,
        "tanimoto": tani(truth, pred),
    })


syn_map = pd.DataFrame(syn_rows)

assert len(syn_map) == 3261
assert syn_map["parent_spec"].nunique() == 333

syn_map.to_csv(
    SYN_MAP,
    index=False
)


with SYN_INPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    for r in syn_map.itertuples(index=False):
        writer.writerow([
            r.pair_id,
            r.truth_smiles,
            r.pred_smiles,
        ])


print()
print("=" * 100)
print("RQ4 SYNTHETIC INPUT")
print("=" * 100)

print("N =", len(syn_map))
print("PARENTS =", syn_map["parent_spec"].nunique())
print("CANDIDATE COLUMN =", candidate_col)

print()
print("PREPARATION PASS")
