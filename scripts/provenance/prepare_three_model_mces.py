from pathlib import Path
import csv
import pandas as pd
from rdkit import Chem

FILES = {
    "DiffMS": Path(r"results\normalized\DIFFMS_K1_EG100.csv"),
    "FRIGID": Path(r"results\normalized\FRIGID_BASE_FM1_EG100_HARMONIZED.csv"),
    "MS2Forge": Path(r"results\normalized\MS2FORGE_K1_EG100.csv"),
}

MANIFEST = Path(r"manifests\MSG_EG100_DEV_v01.csv")

OUT_INPUT = Path(r"results\mces\THREE_MODEL_NEW_PAIRS_INPUT.csv")
OUT_MAP   = Path(r"results\mces\THREE_MODEL_NEW_PAIRS_MAP.csv")
OUT_SPECS = Path(r"results\mces\THREE_MODEL_NEW_COMPARISONS.csv")


def canonical(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        raise RuntimeError(s)

    Chem.RemoveStereochemistry(m)

    return Chem.MolToSmiles(
        m,
        canonical=True,
        isomericSmiles=False
    )


def load(name, path):
    d = pd.read_csv(path)

    return d[
        [
            "panel_index",
            "spec",
            "pred_smiles",
            "task_valid",
            "exact_connectivity",
            "same_formula",
            "tanimoto",
        ]
    ].rename(columns={
        "pred_smiles": f"{name}_pred",
        "task_valid": f"{name}_valid",
        "exact_connectivity": f"{name}_exact",
        "same_formula": f"{name}_formula",
        "tanimoto": f"{name}_tanimoto",
    })


m = pd.read_csv(MANIFEST)[
    ["panel_index", "spec", "canonical_smiles"]
].rename(columns={
    "canonical_smiles": "truth_smiles"
})

x = m.copy()

for name, path in FILES.items():
    x = x.merge(
        load(name, path),
        on=["panel_index", "spec"],
        validate="one_to_one"
    )


comparisons = [
    ("DiffMS", "MS2Forge"),
    ("FRIGID", "MS2Forge"),
]

comparison_rows = []
needed = set()

for A, B in comparisons:

    mask = (
        x[f"{A}_valid"]
        & x[f"{B}_valid"]
        & ~x[f"{A}_exact"]
        & ~x[f"{B}_exact"]
    )

    specs = x.loc[
        mask,
        ["panel_index", "spec"]
    ]

    print(
        f"{A} vs {B}:",
        len(specs)
    )

    for r in specs.itertuples(index=False):

        comparison_rows.append({
            "comparison": f"{A}_vs_{B}",
            "panel_index": r.panel_index,
            "spec": r.spec,
            "model_A": A,
            "model_B": B,
        })

        needed.add((r.spec, A))
        needed.add((r.spec, B))


comp = pd.DataFrame(comparison_rows)

assert (
    len(comp[comp.comparison == "DiffMS_vs_MS2Forge"])
    == 31
)

assert (
    len(comp[comp.comparison == "FRIGID_vs_MS2Forge"])
    == 52
)


records = []
pair_id = 0

lookup = x.set_index("spec")

for spec, model in sorted(needed):

    r = lookup.loc[spec]

    records.append({
        "pair_id": pair_id,
        "panel_index": int(r.panel_index),
        "spec": spec,
        "model": model,
        "same_formula": bool(r[f"{model}_formula"]),
        "tanimoto": float(r[f"{model}_tanimoto"]),
        "truth_smiles": canonical(r.truth_smiles),
        "pred_smiles": canonical(r[f"{model}_pred"]),
    })

    pair_id += 1


out = pd.DataFrame(records)

OUT_INPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with OUT_INPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    for r in out.itertuples(index=False):

        writer.writerow([
            r.pair_id,
            r.truth_smiles,
            r.pred_smiles,
        ])


out.to_csv(OUT_MAP, index=False)
comp.to_csv(OUT_SPECS, index=False)

print()
print("UNIQUE MCES PAIRS =", len(out))
print("INPUT =", OUT_INPUT)
print("MAP   =", OUT_MAP)
print("COMP  =", OUT_SPECS)
