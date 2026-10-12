from pathlib import Path
import csv

import pandas as pd
from rdkit import Chem


REAL = Path(
    r"results\RQ4_REAL_GEOMETRY_EG100.csv"
)

SYN_GEOM = Path(
    r"results\RQ4_SYNTHETIC_GEOMETRY_EG100.csv"
)

SYN_RAW = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_PRIMARY.parquet"
)

REAL_RAW = Path(
    r"results\RQ4_REAL_SAME_FORMULA_WRONG_EG100.csv"
)

OUT_INPUT = Path(
    r"results\mces\RQ4_MATCHED_MCES_INPUT.csv"
)

OUT_MAP = Path(
    r"results\mces\RQ4_MATCHED_MCES_MAP.csv"
)


def canonical(s):

    m = Chem.MolFromSmiles(str(s))

    if m is None:
        raise RuntimeError(s)

    Chem.RemoveStereochemistry(m)

    return Chem.MolToSmiles(
        m,
        canonical=True,
        isomericSmiles=False,
    )


real_geom = pd.read_csv(REAL)
syn_geom = pd.read_csv(SYN_GEOM)

real_raw = pd.read_csv(REAL_RAW)
syn_raw = pd.read_parquet(SYN_RAW)


# ------------------------------------------------------------
# Pick one central synthetic corruption per parent.
# Closest to the parent's median Tanimoto.
# ------------------------------------------------------------

reps = []

for parent, g in syn_geom.groupby(
    "parent_spec"
):

    med = g["tanimoto"].median()

    q = g.copy()

    q["_dist_to_median"] = (
        q["tanimoto"] - med
    ).abs()

    q = q.sort_values(
        [
            "_dist_to_median",
            "observation_id",
        ]
    )

    reps.append(
        q.iloc[0]
    )


rep = pd.DataFrame(reps)


if len(rep) != 88:
    raise RuntimeError(
        f"Expected 88 synthetic reps, got {len(rep)}"
    )


rep = rep.merge(
    syn_raw[
        [
            "sample_id",
            "true_smiles",
            "start_smiles",
            "corruption_family",
        ]
    ],
    left_on="observation_id",
    right_on="sample_id",
    validate="one_to_one",
)


# ------------------------------------------------------------
# Real observations: all 98.
# ------------------------------------------------------------

records = []
pair_id = 0


for r in real_raw.itertuples(index=False):

    records.append({
        "pair_id":
            pair_id,

        "parent_spec":
            r.spec,

        "source":
            "real",

        "subgroup":
            r.model,

        "tanimoto":
            r.tanimoto,

        "truth_smiles":
            canonical(r.truth_smiles),

        "candidate_smiles":
            canonical(r.pred_smiles),
    })

    pair_id += 1


# ------------------------------------------------------------
# Synthetic representative: 88.
# ------------------------------------------------------------

for r in rep.itertuples(index=False):

    records.append({
        "pair_id":
            pair_id,

        "parent_spec":
            r.parent_spec,

        "source":
            "synthetic",

        "subgroup":
            r.corruption_family,

        "tanimoto":
            r.tanimoto,

        "truth_smiles":
            canonical(r.true_smiles),

        "candidate_smiles":
            canonical(r.start_smiles),
    })

    pair_id += 1


out = pd.DataFrame(records)

print("=" * 100)
print("RQ4 MATCHED MCES PREPARATION")
print("=" * 100)

print(
    out["source"]
    .value_counts()
    .to_string()
)

print()
print(
    "Unique real parents =",
    out.loc[
        out["source"] == "real",
        "parent_spec"
    ].nunique()
)

print(
    "Unique synthetic parents =",
    out.loc[
        out["source"] == "synthetic",
        "parent_spec"
    ].nunique()
)


OUT_INPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUT_INPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    for r in out.itertuples(index=False):

        writer.writerow(
            [
                r.pair_id,
                r.truth_smiles,
                r.candidate_smiles,
            ]
        )


out.to_csv(
    OUT_MAP,
    index=False
)

print()
print(
    "TOTAL MCES PAIRS =",
    len(out)
)

print("INPUT =", OUT_INPUT)
print("MAP   =", OUT_MAP)
