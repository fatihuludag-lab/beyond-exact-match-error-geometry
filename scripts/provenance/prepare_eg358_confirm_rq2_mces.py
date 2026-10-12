from pathlib import Path
import csv
import pandas as pd
from rdkit import Chem


INPUT = Path(
    r"results\EG358_CONFIRM_THREE_MODEL_PAIRED.csv"
)

OUT_INPUT = Path(
    r"results\mces\EG358_CONFIRM_RQ2_MCES_INPUT.csv"
)

OUT_MAP = Path(
    r"results\mces\EG358_CONFIRM_RQ2_MCES_MAP.csv"
)

OUT_COMP = Path(
    r"results\mces\EG358_CONFIRM_RQ2_COMPARISONS.csv"
)


def canonical(s):
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


x = pd.read_csv(INPUT)

assert len(x) == 358


pairs = [
    ("DiffMS", "FRIGID"),
    ("DiffMS", "MS2Forge"),
    ("FRIGID", "MS2Forge"),
]


comparison_rows = []
needed = set()


for A, B in pairs:

    q = x[
        x[f"{A}_valid"]
        &
        x[f"{B}_valid"]
        &
        ~x[f"{A}_exact"]
        &
        ~x[f"{B}_exact"]
    ].copy()

    print(
        f"{A} vs {B} =",
        len(q)
    )

    for r in q.itertuples(index=False):

        comparison_rows.append({
            "comparison":
                f"{A}_vs_{B}",

            "spec":
                r.spec,

            "model_A":
                A,

            "model_B":
                B,
        })

        needed.add(
            (r.spec, A)
        )

        needed.add(
            (r.spec, B)
        )


comp = pd.DataFrame(
    comparison_rows
)

assert (
    len(
        comp[
            comp.comparison
            == "DiffMS_vs_FRIGID"
        ]
    )
    == 156
)

assert (
    len(
        comp[
            comp.comparison
            == "DiffMS_vs_MS2Forge"
        ]
    )
    == 94
)

assert (
    len(
        comp[
            comp.comparison
            == "FRIGID_vs_MS2Forge"
        ]
    )
    == 182
)


lookup = (
    x.set_index("spec")
)

rows = []


for pair_id, (spec, model) in enumerate(
    sorted(needed)
):

    r = lookup.loc[spec]

    rows.append({
        "pair_id":
            pair_id,

        "spec":
            spec,

        "model":
            model,

        "same_formula":
            bool(
                r[f"{model}_formula"]
            ),

        "tanimoto":
            float(
                r[f"{model}_tanimoto"]
            ),

        "truth_smiles":
            canonical(
                r["truth_smiles"]
            ),

        "pred_smiles":
            canonical(
                r[f"{model}_pred"]
            ),
    })


out = pd.DataFrame(rows)


print()
print("=" * 90)
print("EG358 CONFIRMATORY RQ2 MCES INPUT")
print("=" * 90)

print(
    "UNIQUE MODEL-SPEC PAIRS =",
    len(out)
)

print(
    "UNIQUE SPECS =",
    out["spec"].nunique()
)

print()
print(
    out["model"]
    .value_counts()
    .to_string()
)


if len(out) != 585:
    raise RuntimeError(
        f"Expected 585 unique MCES pairs, got {len(out)}"
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

        writer.writerow([
            r.pair_id,
            r.truth_smiles,
            r.pred_smiles,
        ])


out.to_csv(
    OUT_MAP,
    index=False
)

comp.to_csv(
    OUT_COMP,
    index=False
)


print()
print("INPUT =", OUT_INPUT)
print("MAP   =", OUT_MAP)
print("COMP  =", OUT_COMP)
print()
print("PREPARATION PASS")
