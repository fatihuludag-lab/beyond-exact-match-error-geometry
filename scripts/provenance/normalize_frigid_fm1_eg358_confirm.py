from pathlib import Path
import json
import pandas as pd
import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


ROOT = Path(r"C:\research\ms_error_geometry")

DETAIL = (
    ROOT / "results" /
    "frigid_base_eg358_confirm_fm1" /
    "detailed_results.csv"
)

MANIFEST = (
    ROOT / "manifests" /
    "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv"
)

OUT = (
    ROOT / "results" /
    "normalized" /
    "FRIGID_BASE_FM1_EG358_CONFIRM_HARMONIZED.csv"
)

SUMMARY = (
    ROOT / "results" /
    "normalized" /
    "FRIGID_BASE_FM1_EG358_CONFIRM_HARMONIZED.summary.json"
)

FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def fresh_mol(s):
    if s is None or pd.isna(s):
        return None

    try:
        m = Chem.MolFromSmiles(
            str(s),
            sanitize=False,
        )

        if m is None:
            return None

        Chem.SanitizeMol(m)
        return m

    except Exception:
        return None


def conn_key(m):
    if m is None:
        return None

    try:
        x = Chem.Mol(m)
        Chem.RemoveStereochemistry(x)

        k = Chem.MolToInchiKey(x)

        return k[:14] if k else None

    except Exception:
        return None


def canonical(m):
    if m is None:
        return None

    try:
        x = Chem.Mol(m)
        Chem.RemoveStereochemistry(x)

        return Chem.MolToSmiles(
            x,
            canonical=True,
            isomericSmiles=False,
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


def components(m):
    if m is None:
        return None

    try:
        return len(
            Chem.GetMolFrags(
                m,
                asMols=False,
                sanitizeFrags=False,
            )
        )
    except Exception:
        return None


def tani(a, b):
    if a is None or b is None:
        return np.nan

    try:
        return float(
            DataStructs.TanimotoSimilarity(
                FPGEN.GetFingerprint(a),
                FPGEN.GetFingerprint(b),
            )
        )
    except Exception:
        return np.nan


detail = pd.read_csv(DETAIL)
manifest = pd.read_csv(MANIFEST)

print("DETAIL COLUMNS:")
print(detail.columns.tolist())
print()
print("DETAIL N =", len(detail))
print("MANIFEST N =", len(manifest))


# ------------------------------------------------------------
# Detect columns robustly
# ------------------------------------------------------------

spec_candidates = [
    "spec",
    "spec_name",
    "spec_id",
    "spectrum_id",
    "name",
    "spectrum",
]

spec_col = next(
    (
        c for c in spec_candidates
        if c in detail.columns
    ),
    None
)

if spec_col is None:
    raise RuntimeError(
        f"Could not identify spectrum ID column: {detail.columns.tolist()}"
    )


if "proposal_smiles" not in detail.columns:
    raise RuntimeError(
        "detailed_results.csv has no proposal_smiles column."
    )


truth_candidates = [
    "smiles",
    "canonical_smiles",
    "truth_smiles",
]

truth_col = next(
    (
        c for c in truth_candidates
        if c in manifest.columns
    ),
    None
)

if truth_col is None:
    raise RuntimeError(
        f"Could not identify manifest truth SMILES: {manifest.columns.tolist()}"
    )


detail[spec_col] = detail[spec_col].astype(str)
manifest["spec"] = manifest["spec"].astype(str)

if detail[spec_col].duplicated().any():
    raise RuntimeError(
        "Duplicate spectrum IDs in detailed_results.csv"
    )

if manifest["spec"].duplicated().any():
    raise RuntimeError(
        "Duplicate spectrum IDs in manifest"
    )


x = manifest.merge(
    detail[
        [
            spec_col,
            "proposal_smiles",
        ]
    ],
    left_on="spec",
    right_on=spec_col,
    how="left",
    validate="one_to_one",
)


if len(x) != 358:
    raise RuntimeError(
        f"Expected 358 merged rows, got {len(x)}"
    )


missing = x[
    "proposal_smiles"
].isna().sum()

print(
    "MISSING proposal_smiles =",
    missing
)


rows = []

for r in x.itertuples(index=False):

    truth_s = getattr(
        r,
        truth_col
    )

    pred_s = getattr(
        r,
        "proposal_smiles"
    )

    truth = fresh_mol(
        truth_s
    )

    if truth is None:
        raise RuntimeError(
            f"Truth failed sanitization: {r.spec}"
        )

    pred = fresh_mol(
        pred_s
    )

    rdkit_valid = (
        pred is not None
    )

    n_comp = (
        components(pred)
        if pred is not None
        else None
    )

    single_component = bool(
        rdkit_valid
        and n_comp == 1
    )

    task_valid = single_component

    truth_formula = formula(
        truth
    )

    pred_formula = formula(
        pred
    )

    same_formula = bool(
        task_valid
        and pred_formula == truth_formula
    )

    truth_key = conn_key(
        truth
    )

    pred_key = (
        conn_key(pred)
        if pred is not None
        else None
    )

    exact = bool(
        task_valid
        and pred_key is not None
        and pred_key == truth_key
    )

    rows.append({
        "model":
            "FRIGID-base@FM1",

        "panel_index":
            int(r.panel_index),

        "common_support_index":
            (
                int(r.common_support_index)
                if hasattr(r, "common_support_index")
                else None
            ),

        "spec":
            str(r.spec),

        "truth_smiles":
            canonical(truth),

        "pred_smiles":
            canonical(pred),

        "truth_formula":
            truth_formula,

        "pred_formula":
            pred_formula,

        "truth_connectivity_key":
            truth_key,

        "pred_connectivity_key":
            pred_key,

        "rdkit_valid":
            rdkit_valid,

        "n_components":
            n_comp,

        "single_component":
            single_component,

        "task_valid":
            task_valid,

        "same_formula":
            same_formula,

        "exact_connectivity":
            exact,

        "tanimoto":
            tani(
                truth,
                pred
            ),
    })


out = pd.DataFrame(rows)

out = (
    out.sort_values(
        "common_support_index"
    )
    .reset_index(drop=True)
)

assert len(out) == 358
assert out["spec"].nunique() == 358


wrong = out[
    ~out["exact_connectivity"]
]

valid_wrong = out[
    out["task_valid"]
    &
    ~out["exact_connectivity"]
]


summary = {
    "model":
        "FRIGID-base@FM1",

    "panel":
        "MSG-EG358-COMMON-SUPPORT-CONFIRM-v01",

    "n":
        int(len(out)),

    "rdkit_valid":
        int(out["rdkit_valid"].sum()),

    "single_component":
        int(out["single_component"].sum()),

    "task_valid":
        int(out["task_valid"].sum()),

    "same_formula":
        int(out["same_formula"].sum()),

    "exact":
        int(out["exact_connectivity"].sum()),

    "wrong":
        int(len(wrong)),

    "task_valid_wrong":
        int(len(valid_wrong)),

    "mean_tanimoto_task_valid":
        (
            float(
                out.loc[
                    out["task_valid"],
                    "tanimoto"
                ].mean()
            )
            if out["task_valid"].any()
            else None
        ),

    "mean_tanimoto_wrong_task_valid":
        (
            float(
                valid_wrong[
                    "tanimoto"
                ].mean()
            )
            if len(valid_wrong)
            else None
        ),

    "median_tanimoto_wrong_task_valid":
        (
            float(
                valid_wrong[
                    "tanimoto"
                ].median()
            )
            if len(valid_wrong)
            else None
        ),
}


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    OUT,
    index=False
)

SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


print()
print("=" * 100)
print("FRIGID BASE FM1 EG358 CONFIRM — STRICT HARMONIZATION")
print("=" * 100)

for k, v in summary.items():
    print(f"{k} = {v}")

print()
print("NATIVE EXACT ≈ 10 / 358")
print("NATIVE TANIMOTO TOP1 = 0.3366")
print("NATIVE FORMULA SUCCESS ≈ 344 / 358")

print()
print("CSV =", OUT)
print("SUMMARY =", SUMMARY)
print()
print("FRIGID EG358 STRICT HARMONIZATION PASS")

