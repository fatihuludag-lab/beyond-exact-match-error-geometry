from pathlib import Path
import csv
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


REAL = Path(
    r"results\RQ4_V2_REAL_THREE_MODELS.csv"
)

NEW_SYN = Path(
    r"results\RQ4_V2_SYNTHETIC_NEW2_RAW.parquet"
)

OUT_INPUT = Path(
    r"results\mces\RQ4_V2_INCREMENTAL_41_INPUT.csv"
)

OUT_MAP = Path(
    r"results\mces\RQ4_V2_INCREMENTAL_41_MAP.csv"
)


FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def get_mol(s):
    m = Chem.MolFromSmiles(str(s))

    if m is None:
        raise RuntimeError(
            f"Invalid SMILES: {s}"
        )

    Chem.RemoveStereochemistry(m)

    return m


def canonical(s):
    return Chem.MolToSmiles(
        get_mol(s),
        canonical=True,
        isomericSmiles=False,
    )


def tani(a, b):
    a = get_mol(a)
    b = get_mol(b)

    return float(
        DataStructs.TanimotoSimilarity(
            FPGEN.GetFingerprint(a),
            FPGEN.GetFingerprint(b),
        )
    )


real = pd.read_csv(REAL)

# Only the newly added real observations.
real = real[
    real["model"] == "MS2Forge-K1"
].copy()

if len(real) != 21:
    raise RuntimeError(
        f"Expected 21 MS2Forge errors, got {len(real)}"
    )


syn = pd.read_parquet(NEW_SYN)

syn["parent_spec"] = (
    syn["sample_id"]
    .astype(str)
    .str.replace(
        r"__\d+$",
        "",
        regex=True
    )
)

syn = syn[
    syn["inverse_verified"]
    &
    ~syn["inverse_truncated_same_family"]
    &
    ~syn["forward_truncated_family"]
].copy()

if len(syn) != 20:
    raise RuntimeError(
        f"Expected 20 new synthetic errors, got {len(syn)}"
    )

if syn["parent_spec"].nunique() != 2:
    raise RuntimeError(
        "Expected exactly 2 new synthetic parents"
    )


rows = []
pair_id = 0


# ------------------------------------------------------------
# 21 new MS2Forge real errors
# ------------------------------------------------------------

for r in real.itertuples(index=False):

    rows.append({
        "pair_id":
            pair_id,

        "source":
            "real",

        "parent_spec":
            r.parent_spec,

        "observation_id":
            f"MS2Forge-K1:{r.parent_spec}",

        "subgroup":
            "MS2Forge-K1",

        "tanimoto":
            float(r.tanimoto),

        "truth_smiles":
            canonical(r.truth_smiles),

        "candidate_smiles":
            canonical(r.pred_smiles),
    })

    pair_id += 1


# ------------------------------------------------------------
# 20 synthetic errors for the two new parents
# ------------------------------------------------------------

for r in syn.itertuples(index=False):

    rows.append({
        "pair_id":
            pair_id,

        "source":
            "synthetic",

        "parent_spec":
            r.parent_spec,

        "observation_id":
            r.sample_id,

        "subgroup":
            r.corruption_family,

        "tanimoto":
            tani(
                r.true_smiles,
                r.start_smiles,
            ),

        "truth_smiles":
            canonical(r.true_smiles),

        "candidate_smiles":
            canonical(r.start_smiles),
    })

    pair_id += 1


out = pd.DataFrame(rows)

if len(out) != 41:
    raise RuntimeError(
        f"Expected 41 pairs, got {len(out)}"
    )


OUT_INPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUT_INPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as fh:

    writer = csv.writer(fh)

    for r in out.itertuples(index=False):

        writer.writerow([
            r.pair_id,
            r.truth_smiles,
            r.candidate_smiles,
        ])


out.to_csv(
    OUT_MAP,
    index=False
)


print("=" * 90)
print("RQ4 V2 INCREMENTAL MCES")
print("=" * 90)

print(
    out["source"]
    .value_counts()
    .to_string()
)

print()
print(
    "REAL parents =",
    out.loc[
        out.source == "real",
        "parent_spec"
    ].nunique()
)

print(
    "SYN parents =",
    out.loc[
        out.source == "synthetic",
        "parent_spec"
    ].nunique()
)

print()
print("TOTAL =", len(out))
print("INPUT =", OUT_INPUT)
print("MAP   =", OUT_MAP)
