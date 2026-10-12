from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


ROOT = Path(r"C:\research\ms_error_geometry")

PRED_FILE = (
    ROOT / "results" /
    "diffms_eg358_confirm_k1_nw0" /
    "preds" /
    "MSG_EG358_CONFIRM_DIFFMS_K1_ALL358_pred.pkl"
)

TRUE_FILE = (
    ROOT / "results" /
    "diffms_eg358_confirm_k1_nw0" /
    "preds" /
    "MSG_EG358_CONFIRM_DIFFMS_K1_ALL358_true.pkl"
)

MANIFEST = (
    ROOT / "manifests" /
    "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv"
)

OUT = (
    ROOT / "results" /
    "normalized" /
    "DIFFMS_K1_EG358_CONFIRM.csv"
)

SUMMARY = (
    ROOT / "results" /
    "normalized" /
    "DIFFMS_K1_EG358_CONFIRM.summary.json"
)

FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def fresh_sanitize(m):
    if m is None:
        return None

    try:
        x = Chem.Mol(m)
        Chem.SanitizeMol(x)
        return x
    except Exception:
        return None


def connectivity_key(m):
    if m is None:
        return None

    try:
        x = Chem.Mol(m)
        Chem.RemoveStereochemistry(x)

        k = Chem.MolToInchiKey(x)

        return k[:14] if k else None
    except Exception:
        return None


def canonical_smiles(m):
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


def mol_formula(m):
    if m is None:
        return None

    try:
        return CalcMolFormula(m)
    except Exception:
        return None


def n_components(m):
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


def tanimoto(a, b):
    if a is None or b is None:
        return np.nan

    try:
        fa = FPGEN.GetFingerprint(a)
        fb = FPGEN.GetFingerprint(b)

        return float(
            DataStructs.TanimotoSimilarity(
                fa,
                fb,
            )
        )
    except Exception:
        return np.nan


# ------------------------------------------------------------------
# Load canonical persisted artifacts
# ------------------------------------------------------------------

with PRED_FILE.open("rb") as f:
    preds = pickle.load(f)

with TRUE_FILE.open("rb") as f:
    truths = pickle.load(f)

print("PRED LEN =", len(preds))
print("TRUE LEN =", len(truths))

assert len(preds) == 358
assert len(truths) == 358
assert all(len(x) == 1 for x in preds)


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------

manifest = pd.read_csv(MANIFEST)

assert len(manifest) == 358
assert manifest["spec"].nunique() == 358

if "connectivity_key" not in manifest.columns:
    raise RuntimeError(
        "Manifest has no connectivity_key column."
    )

manifest["connectivity_key"] = (
    manifest["connectivity_key"]
    .astype(str)
    .str[:14]
)

assert (
    manifest["connectivity_key"].nunique()
    == 358
)

manifest_by_key = (
    manifest
    .set_index("connectivity_key")
)


# ------------------------------------------------------------------
# Strict fresh-RDKit audit
# ------------------------------------------------------------------

rows = []
manifest_matches = 0
sanitize_failures = 0


for i, (pred_list, truth_raw) in enumerate(
    zip(preds, truths)
):

    truth = fresh_sanitize(
        truth_raw
    )

    if truth is None:
        raise RuntimeError(
            f"Truth failed fresh sanitization at index {i}"
        )

    truth_key = connectivity_key(
        truth
    )

    if truth_key not in manifest_by_key.index:
        raise RuntimeError(
            f"Truth-manifest mismatch at index {i}: {truth_key}"
        )

    meta = manifest_by_key.loc[
        truth_key
    ]

    manifest_matches += 1

    pred_raw = (
        pred_list[0]
        if pred_list
        else None
    )

    pred = fresh_sanitize(
        pred_raw
    )

    rdkit_valid = (
        pred is not None
    )

    if not rdkit_valid:
        sanitize_failures += 1

    comp = (
        n_components(pred)
        if rdkit_valid
        else None
    )

    single_component = bool(
        rdkit_valid
        and comp == 1
    )

    # Common task-valid definition used across generators.
    task_valid = single_component

    truth_formula = mol_formula(
        truth
    )

    pred_formula = mol_formula(
        pred
    )

    same_formula = bool(
        task_valid
        and pred_formula == truth_formula
    )

    pred_key = (
        connectivity_key(pred)
        if rdkit_valid
        else None
    )

    exact = bool(
        task_valid
        and pred_key is not None
        and pred_key == truth_key
    )

    rows.append({
        "model":
            "DiffMS-K1",

        "panel_index":
            int(meta["panel_index"]),

        "common_support_index":
            (
                int(meta["common_support_index"])
                if "common_support_index" in meta.index
                else None
            ),

        "spec":
            str(meta["spec"]),

        "truth_smiles":
            canonical_smiles(truth),

        "pred_smiles":
            canonical_smiles(pred),

        "truth_connectivity_key":
            truth_key,

        "pred_connectivity_key":
            pred_key,

        "truth_formula":
            truth_formula,

        "pred_formula":
            pred_formula,

        "rdkit_valid":
            rdkit_valid,

        "n_components":
            comp,

        "single_component":
            single_component,

        "task_valid":
            task_valid,

        "same_formula":
            same_formula,

        "exact_connectivity":
            exact,

        "tanimoto":
            tanimoto(
                truth,
                pred
            ),
    })


out = pd.DataFrame(rows)

if "common_support_index" in out.columns:
    out = (
        out.sort_values(
            "common_support_index"
        )
        .reset_index(drop=True)
    )

assert len(out) == 358
assert out["spec"].nunique() == 358
assert manifest_matches == 358


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

wrong = out[
    ~out["exact_connectivity"]
].copy()

valid_wrong = out[
    out["task_valid"]
    &
    ~out["exact_connectivity"]
].copy()


summary = {
    "model":
        "DiffMS-K1",

    "panel":
        "MSG-EG358-COMMON-SUPPORT-CONFIRM-v01",

    "n":
        int(len(out)),

    "manifest_key_agreement":
        int(manifest_matches),

    "fresh_sanitize_failures":
        int(sanitize_failures),

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
                valid_wrong["tanimoto"].mean()
            )
            if len(valid_wrong)
            else None
        ),

    "median_tanimoto_wrong_task_valid":
        (
            float(
                valid_wrong["tanimoto"].median()
            )
            if len(valid_wrong)
            else None
        ),
}


OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT,
    index=False,
)

SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 100)
print("DIFFMS K1 EG358 CONFIRM — STRICT NORMALIZATION")
print("=" * 100)

for k, v in summary.items():
    print(f"{k} = {v}")

print()
print(
    "RAW MODEL VALIDITY =",
    0.6955307126045227
)

print(
    "RAW MODEL TANIMOTO =",
    0.11383680254220963
)

print()
print("CSV =", OUT)
print("SUMMARY =", SUMMARY)
print()
print("DIFFMS EG358 STRICT NORMALIZATION PASS")
