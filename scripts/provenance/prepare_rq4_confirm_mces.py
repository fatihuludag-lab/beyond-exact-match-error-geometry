from pathlib import Path
import csv

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(r"C:\research\ms_error_geometry")

REAL = (
    ROOT / "results" /
    "RQ4_CONFIRM_REAL_THREE_MODELS_EG358.csv"
)

SYN = (
    ROOT / "results" /
    "RQ4_CONFIRM_SYNTHETIC_PRIMARY_EG358.parquet"
)

RQ2_MAP = (
    ROOT / "results" / "mces" /
    "EG358_CONFIRM_RQ2_MCES_MAP.csv"
)

RQ2_OUT = (
    ROOT / "results" / "mces" /
    "EG358_CONFIRM_RQ2_MCES_OUT.csv"
)

OUT_REAL_REUSED = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_REUSED_FROM_RQ2.csv"
)

OUT_REAL_MISSING_MAP = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_MISSING_MAP.csv"
)

OUT_REAL_MISSING_INPUT = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_MISSING_INPUT.csv"
)

OUT_SYN_MAP = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_SYNTHETIC_MAP.csv"
)

OUT_SYN_INPUT = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_SYNTHETIC_INPUT.csv"
)


FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


MODEL_ALIAS = {
    "DiffMS-K1": "DiffMS",
    "FRIGID-base@FM1": "FRIGID",
    "MS2Forge-K1": "MS2Forge",
}


def mol(s):
    m = Chem.MolFromSmiles(str(s))

    if m is None:
        raise RuntimeError(
            f"Invalid SMILES: {s}"
        )

    Chem.RemoveStereochemistry(m)

    return m


def canonical(s):
    return Chem.MolToSmiles(
        mol(s),
        canonical=True,
        isomericSmiles=False,
    )


def tani(a, b):
    ma = mol(a)
    mb = mol(b)

    return float(
        DataStructs.TanimotoSimilarity(
            FPGEN.GetFingerprint(ma),
            FPGEN.GetFingerprint(mb),
        )
    )


# ============================================================
# Existing exact MCES from confirmatory RQ2
# ============================================================

rq2_map = pd.read_csv(RQ2_MAP)

rq2_dist = pd.read_csv(
    RQ2_OUT,
    header=None,
    names=[
        "pair_id",
        "mces",
        "time_sec",
        "mode",
    ],
)

assert len(rq2_map) == 585
assert len(rq2_dist) == 585
assert set(rq2_dist["mode"]) == {1}
assert not (rq2_dist["mces"] < 0).any()


rq2 = rq2_map.merge(
    rq2_dist,
    on="pair_id",
    validate="one_to_one",
)

assert not rq2.duplicated(
    ["spec", "model"]
).any()


# ============================================================
# REAL — 430 same-formula wrong observations
# ============================================================

real = pd.read_csv(REAL)

assert len(real) == 430
assert real["spec"].nunique() == 333

real["rq2_model"] = (
    real["model"]
    .map(MODEL_ALIAS)
)

if real["rq2_model"].isna().any():
    raise RuntimeError(
        "Unknown model alias."
    )


rq2_lookup = (
    rq2.set_index(
        ["spec", "model"]
    )
)


reused = []
missing = []

representation_mismatch = 0


for r in real.itertuples(index=False):

    key = (
        str(r.spec),
        str(r.rq2_model),
    )

    if key in rq2_lookup.index:

        old = rq2_lookup.loc[key]

        # Hard audit before reusing MCES.
        if canonical(r.pred_smiles) != canonical(
            old["pred_smiles"]
        ):
            raise RuntimeError(
                f"Prediction mismatch for {key}"
            )

        if canonical(r.truth_smiles) != canonical(
            old["truth_smiles"]
        ):
            representation_mismatch += 1

            # Canonical connectivity must still agree.
            if canonical(r.truth_smiles) != canonical(
                old["truth_smiles"]
            ):
                raise RuntimeError(
                    f"Truth mismatch for {key}"
                )

        reused.append({
            "model":
                r.model,

            "spec":
                r.spec,

            "parent_spec":
                r.spec,

            "truth_smiles":
                canonical(r.truth_smiles),

            "pred_smiles":
                canonical(r.pred_smiles),

            "tanimoto":
                float(r.tanimoto),

            "mces":
                float(old["mces"]),

            "mces_source":
                "RQ2_exact_reuse",
        })

    else:

        missing.append({
            "model":
                r.model,

            "spec":
                r.spec,

            "parent_spec":
                r.spec,

            "truth_smiles":
                canonical(r.truth_smiles),

            "pred_smiles":
                canonical(r.pred_smiles),

            "tanimoto":
                float(r.tanimoto),
        })


reused = pd.DataFrame(reused)
missing = pd.DataFrame(missing)


print("=" * 100)
print("RQ4 CONFIRMATORY REAL MCES REUSE AUDIT")
print("=" * 100)

print(
    "REAL TOTAL =",
    len(real)
)

print(
    "REUSED EXACT MCES FROM RQ2 =",
    len(reused)
)

print(
    "NEW REAL MCES REQUIRED =",
    len(missing)
)

print(
    "REUSE + MISSING =",
    len(reused) + len(missing)
)

assert (
    len(reused) + len(missing)
    == 430
)


reused.to_csv(
    OUT_REAL_REUSED,
    index=False,
)


# Assign independent pair IDs for missing real.
if len(missing):

    missing = (
        missing.reset_index(drop=True)
    )

    missing["pair_id"] = range(
        len(missing)
    )

    missing.to_csv(
        OUT_REAL_MISSING_MAP,
        index=False,
    )

    with OUT_REAL_MISSING_INPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        for r in missing.itertuples(index=False):

            writer.writerow([
                r.pair_id,
                r.truth_smiles,
                r.pred_smiles,
            ])


# ============================================================
# SYNTHETIC — 3261 primary corruptions
# ============================================================

syn = pd.read_parquet(SYN)

assert len(syn) == 3261
assert syn["parent_spec"].nunique() == 333


candidate_cols = [
    "start_smiles",
    "corrupted_smiles",
    "candidate_smiles",
    "pred_smiles",
]

candidate_col = next(
    (
        c
        for c in candidate_cols
        if c in syn.columns
    ),
    None
)

if candidate_col is None:
    raise RuntimeError(
        "Cannot identify synthetic candidate SMILES. "
        f"Columns={syn.columns.tolist()}"
    )


syn_rows = []

for pair_id, r in enumerate(
    syn.itertuples(index=False)
):

    truth = canonical(
        r.true_smiles
    )

    candidate = canonical(
        getattr(
            r,
            candidate_col
        )
    )

    syn_rows.append({
        "pair_id":
            pair_id,

        "sample_id":
            r.sample_id,

        "parent_spec":
            r.parent_spec,

        "corruption_family":
            r.corruption_family,

        "truth_smiles":
            truth,

        "pred_smiles":
            candidate,

        "tanimoto":
            tani(
                truth,
                candidate,
            ),
    })


syn_map = pd.DataFrame(
    syn_rows
)

assert len(syn_map) == 3261
assert syn_map["parent_spec"].nunique() == 333


syn_map.to_csv(
    OUT_SYN_MAP,
    index=False,
)


with OUT_SYN_INPUT.open(
    "w",
    newline="",
    encoding="utf-8",
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
print("RQ4 CONFIRMATORY SYNTHETIC MCES INPUT")
print("=" * 100)

print(
    "SYNTHETIC N =",
    len(syn_map)
)

print(
    "PARENTS =",
    syn_map["parent_spec"].nunique()
)

print(
    "CANDIDATE COLUMN =",
    candidate_col
)

print()
print(
    syn_map["corruption_family"]
    .value_counts()
    .to_string()
)

print()
print("REAL REUSED =", OUT_REAL_REUSED)
print("REAL MISSING MAP =", OUT_REAL_MISSING_MAP)
print("REAL MISSING INPUT =", OUT_REAL_MISSING_INPUT)
print("SYN MAP =", OUT_SYN_MAP)
print("SYN INPUT =", OUT_SYN_INPUT)

print()
print("RQ4 CONFIRMATORY MCES PREPARATION PASS")
