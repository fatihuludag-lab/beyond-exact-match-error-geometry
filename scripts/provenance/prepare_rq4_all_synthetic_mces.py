from pathlib import Path
import csv

import pandas as pd
from rdkit import Chem


RAW = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_PRIMARY.parquet"
)

GEOM = Path(
    r"results\RQ4_SYNTHETIC_GEOMETRY_EG100.csv"
)

OUT_INPUT = Path(
    r"results\mces\RQ4_SYNTHETIC_ALL860_INPUT.csv"
)

OUT_MAP = Path(
    r"results\mces\RQ4_SYNTHETIC_ALL860_MAP.csv"
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


raw = pd.read_parquet(RAW)

geom = pd.read_csv(GEOM)

x = raw.merge(
    geom[
        [
            "observation_id",
            "tanimoto",
        ]
    ],
    left_on="sample_id",
    right_on="observation_id",
    validate="one_to_one",
)


assert len(x) == 860
assert x["parent_spec"].nunique() == 88


rows = []

for pair_id, r in enumerate(
    x.itertuples(index=False)
):

    rows.append({
        "pair_id":
            pair_id,

        "parent_spec":
            r.parent_spec,

        "sample_id":
            r.sample_id,

        "corruption_family":
            r.corruption_family,

        "tanimoto":
            r.tanimoto,

        "truth_smiles":
            canonical(r.true_smiles),

        "candidate_smiles":
            canonical(r.start_smiles),
    })


out = pd.DataFrame(rows)

OUT_INPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)


with OUT_INPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    w = csv.writer(f)

    for r in out.itertuples(index=False):

        w.writerow([
            r.pair_id,
            r.truth_smiles,
            r.candidate_smiles,
        ])


out.to_csv(
    OUT_MAP,
    index=False
)


print("N =", len(out))
print(
    "PARENTS =",
    out["parent_spec"].nunique()
)

print()
print(
    out["corruption_family"]
    .value_counts()
    .to_string()
)

print()
print("INPUT =", OUT_INPUT)
print("MAP =", OUT_MAP)

