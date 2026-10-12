from pathlib import Path
import csv

import pandas as pd
from rdkit import Chem


DIFFMS = Path(
    r"results\normalized\DIFFMS_K1_EG100.csv"
)

FRIGID = Path(
    r"results\normalized\FRIGID_BASE_FM1_EG100_HARMONIZED.csv"
)

MANIFEST = Path(
    r"manifests\MSG_EG100_DEV_v01.csv"
)

OUT_INPUT = Path(
    r"results\mces\EG100_COMMON_WRONG_MCES_INPUT.csv"
)

OUT_MAP = Path(
    r"results\mces\EG100_COMMON_WRONG_MCES_MAP.csv"
)

OUT_SMOKE = Path(
    r"results\mces\EG100_COMMON_WRONG_MCES_SMOKE.csv"
)


def canonical_nostereo(s):
    m = Chem.MolFromSmiles(str(s))

    if m is None:
        raise RuntimeError(
            f"Invalid SMILES: {s}"
        )

    Chem.RemoveStereochemistry(m)

    return Chem.MolToSmiles(
        m,
        canonical=True,
        isomericSmiles=False,
    )


d = pd.read_csv(DIFFMS)
f = pd.read_csv(FRIGID)
m = pd.read_csv(MANIFEST)

truth = (
    m[
        [
            "panel_index",
            "spec",
            "canonical_smiles",
        ]
    ]
    .rename(
        columns={
            "canonical_smiles":
                "truth_smiles"
        }
    )
)


d = d[
    [
        "panel_index",
        "spec",
        "pred_smiles",
        "task_valid",
        "exact_connectivity",
        "same_formula",
        "tanimoto",
    ]
].rename(
    columns={
        "pred_smiles":
            "diffms_pred",

        "task_valid":
            "diffms_valid",

        "exact_connectivity":
            "diffms_exact",

        "same_formula":
            "diffms_formula",

        "tanimoto":
            "diffms_tanimoto",
    }
)


f = f[
    [
        "panel_index",
        "spec",
        "pred_smiles",
        "task_valid",
        "exact_connectivity",
        "same_formula",
        "tanimoto",
    ]
].rename(
    columns={
        "pred_smiles":
            "frigid_pred",

        "task_valid":
            "frigid_valid",

        "exact_connectivity":
            "frigid_exact",

        "same_formula":
            "frigid_formula",

        "tanimoto":
            "frigid_tanimoto",
    }
)


x = (
    truth
    .merge(
        d,
        on=["panel_index", "spec"],
        validate="one_to_one",
    )
    .merge(
        f,
        on=["panel_index", "spec"],
        validate="one_to_one",
    )
)


x = x[
    x["diffms_valid"]
    &
    x["frigid_valid"]
    &
    ~x["diffms_exact"]
    &
    ~x["frigid_exact"]
].copy()


if len(x) != 46:
    raise RuntimeError(
        f"Expected 46 common-valid-wrong, got {len(x)}"
    )


rows = []
pair_id = 0


for r in x.itertuples(index=False):

    truth_smi = canonical_nostereo(
        r.truth_smiles
    )

    for (
        model,
        pred_smi,
        same_formula,
        tani,
    ) in [
        (
            "DiffMS-K1",
            r.diffms_pred,
            r.diffms_formula,
            r.diffms_tanimoto,
        ),
        (
            "FRIGID-base@FM1",
            r.frigid_pred,
            r.frigid_formula,
            r.frigid_tanimoto,
        ),
    ]:

        pred_smi = canonical_nostereo(
            pred_smi
        )

        rows.append({
            "pair_id":
                pair_id,

            "panel_index":
                int(r.panel_index),

            "spec":
                r.spec,

            "model":
                model,

            "same_formula":
                bool(same_formula),

            "tanimoto":
                float(tani),

            "truth_smiles":
                truth_smi,

            "pred_smiles":
                pred_smi,
        })

        pair_id += 1


out = pd.DataFrame(rows)

if len(out) != 92:
    raise RuntimeError(
        f"Expected 92 pairs, got {len(out)}"
    )


OUT_INPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# myopic-mces input has NO HEADER:
# index,SMILES1,SMILES2

with OUT_INPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as fh:

    writer = csv.writer(fh)

    for r in out.itertuples(index=False):

        writer.writerow(
            [
                r.pair_id,
                r.truth_smiles,
                r.pred_smiles,
            ]
        )


out.to_csv(
    OUT_MAP,
    index=False,
)


# First 8 pairs = 4 molecules × 2 models.
with OUT_SMOKE.open(
    "w",
    newline="",
    encoding="utf-8",
) as fh:

    writer = csv.writer(fh)

    for r in out.head(8).itertuples(
        index=False
    ):

        writer.writerow(
            [
                r.pair_id,
                r.truth_smiles,
                r.pred_smiles,
            ]
        )


print("=" * 100)
print("MCES INPUT PREPARATION")
print("=" * 100)

print(
    "common-valid-wrong molecules =",
    x["spec"].nunique()
)

print(
    "total MCES pairs =",
    len(out)
)

print()
print(
    out["model"]
    .value_counts()
    .to_string()
)

print()
print(
    "same-formula pairs by model:"
)

print(
    out.groupby("model")[
        "same_formula"
    ].sum().to_string()
)

print()
print("INPUT =", OUT_INPUT)
print("MAP   =", OUT_MAP)
print("SMOKE =", OUT_SMOKE)

print()
print("MCES PREPARATION PASS")
