from pathlib import Path
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


PRED = Path(
    r"results\frigid_base_fm10_smoke10\predictions.csv"
)

MANIFEST = Path(
    r"manifests\MSG_EG100_DEV_v01.csv"
)

TOP1 = Path(
    r"results\normalized\FRIGID_BASE_FM10_SMOKE10.csv"
)

OUT_CAND = Path(
    r"results\normalized\FRIGID_BASE_FM10_SMOKE10_CANDIDATES.csv"
)

OUT_SUM = Path(
    r"results\normalized\FRIGID_BASE_FM10_SMOKE10_POOL_SUMMARY.csv"
)


fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def mol_from_smiles(s):
    if pd.isna(s):
        return None

    s = str(s).strip()

    if not s:
        return None

    try:
        m = Chem.MolFromSmiles(
            s,
            sanitize=False
        )

        if m is None:
            return None

        Chem.SanitizeMol(m)

        return Chem.Mol(m)

    except Exception:
        return None


def conn_key(m):
    if m is None:
        return None

    try:
        x = Chem.Mol(m)
        Chem.RemoveStereochemistry(x)

        key = Chem.MolToInchiKey(x)

        return (
            key[:14]
            if key
            else None
        )

    except Exception:
        return None


def formula(m):
    if m is None:
        return None

    try:
        return CalcMolFormula(m)
    except Exception:
        return None


def tani(a, b):
    if a is None or b is None:
        return None

    fa = fpgen.GetFingerprint(a)
    fb = fpgen.GetFingerprint(b)

    return float(
        DataStructs.TanimotoSimilarity(
            fa, fb
        )
    )


pred = pd.read_csv(PRED)
manifest = pd.read_csv(MANIFEST)
top1 = pd.read_csv(TOP1)

manifest = manifest.set_index("spec")
top1 = top1.set_index("spec")

pred_cols = [
    c
    for c in pred.columns
    if c.startswith("pred_smiles_")
]

pred_cols = sorted(
    pred_cols,
    key=lambda c: int(
        c.rsplit("_", 1)[1]
    )
)

rows = []

for r in pred.itertuples(index=False):

    spec = str(r.name)

    if spec not in manifest.index:
        raise RuntimeError(
            f"{spec} missing from manifest"
        )

    truth = mol_from_smiles(
        manifest.loc[spec, "smiles"]
    )

    if truth is None:
        raise RuntimeError(
            f"Invalid truth: {spec}"
        )

    tf = formula(truth)
    tk = conn_key(truth)

    for rank, col in enumerate(
        pred_cols,
        start=1
    ):

        smi = getattr(r, col)

        if pd.isna(smi) or not str(smi).strip():
            continue

        m = mol_from_smiles(smi)

        valid = m is not None

        pf = formula(m)
        pk = conn_key(m)

        rows.append({
            "spec": spec,
            "saved_position": rank,

            "candidate_smiles":
                str(smi),

            "valid":
                valid,

            "truth_formula":
                tf,

            "pred_formula":
                pf,

            "same_formula":
                bool(
                    valid
                    and pf == tf
                ),

            "exact_connectivity":
                bool(
                    valid
                    and pk is not None
                    and pk == tk
                ),

            "truth_tanimoto":
                tani(
                    truth,
                    m
                ),
        })


cand = pd.DataFrame(rows)

cand.to_csv(
    OUT_CAND,
    index=False
)


summary_rows = []

for spec, g in cand.groupby("spec"):

    formula_g = g[
        g["same_formula"]
    ]

    valid_g = g[
        g["valid"]
    ]

    exact_any = bool(
        g["exact_connectivity"].any()
    )

    top = top1.loc[spec]

    summary_rows.append({
        "spec":
            spec,

        "n_saved_candidates":
            len(g),

        "n_valid_candidates":
            int(
                g["valid"].sum()
            ),

        "n_strict_formula_candidates":
            int(
                g["same_formula"].sum()
            ),

        "n_nonformula_candidates":
            int(
                (
                    g["valid"]
                    &
                    ~g["same_formula"]
                ).sum()
            ),

        "exact_anywhere_in_pool":
            exact_any,

        "model_top1_exact":
            bool(
                top[
                    "exact_connectivity"
                ]
            ),

        "model_top1_same_formula":
            bool(
                top[
                    "same_formula"
                ]
            ),

        "model_top1_tanimoto":
            float(
                top["tanimoto"]
            ),

        "oracle_best_tanimoto_all":
            (
                float(
                    valid_g[
                        "truth_tanimoto"
                    ].max()
                )
                if len(valid_g)
                else None
            ),

        "oracle_best_tanimoto_formula":
            (
                float(
                    formula_g[
                        "truth_tanimoto"
                    ].max()
                )
                if len(formula_g)
                else None
            ),
    })


summary = pd.DataFrame(
    summary_rows
).sort_values("spec")

summary[
    "oracle_gain_all"
] = (
    summary[
        "oracle_best_tanimoto_all"
    ]
    -
    summary[
        "model_top1_tanimoto"
    ]
)

summary[
    "oracle_gain_formula"
] = (
    summary[
        "oracle_best_tanimoto_formula"
    ]
    -
    summary[
        "model_top1_tanimoto"
    ]
)

summary.to_csv(
    OUT_SUM,
    index=False
)


print("=" * 110)
print("FRIGID FM10 STRICT CANDIDATE-POOL AUDIT")
print("=" * 110)

print(
    summary.to_string(
        index=False
    )
)

print()
print("=" * 110)
print("SUMMARY")
print("=" * 110)

print(
    "spectra =",
    len(summary)
)

print(
    "mean saved candidates =",
    summary[
        "n_saved_candidates"
    ].mean()
)

print(
    "mean strict formula candidates =",
    summary[
        "n_strict_formula_candidates"
    ].mean()
)

print(
    "spectra with <10 strict formula candidates =",
    int(
        (
            summary[
                "n_strict_formula_candidates"
            ] < 10
        ).sum()
    ),
    "/",
    len(summary)
)

print(
    "Top1 strict formula =",
    int(
        summary[
            "model_top1_same_formula"
        ].sum()
    ),
    "/",
    len(summary)
)

print(
    "Exact Top1 =",
    int(
        summary[
            "model_top1_exact"
        ].sum()
    ),
    "/",
    len(summary)
)

print(
    "Exact anywhere in saved pool =",
    int(
        summary[
            "exact_anywhere_in_pool"
        ].sum()
    ),
    "/",
    len(summary)
)

print()
print(
    "Mean model Top1 Tanimoto =",
    summary[
        "model_top1_tanimoto"
    ].mean()
)

print(
    "Mean oracle pool Tanimoto =",
    summary[
        "oracle_best_tanimoto_all"
    ].mean()
)

print(
    "Mean oracle formula-only Tanimoto =",
    summary[
        "oracle_best_tanimoto_formula"
    ].mean()
)

print(
    "Mean oracle gain =",
    summary[
        "oracle_gain_all"
    ].mean()
)

print()
print("Candidates =", OUT_CAND)
print("Summary    =", OUT_SUM)
