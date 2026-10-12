from pathlib import Path
import pickle
import json

import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


# =============================================================================
# PATHS
# =============================================================================

PRED_PKL = Path(
    r"C:\research\ms_error_geometry\results"
    r"\diffms_eg100_k1\preds"
    r"\MSG_EG100_DIFFMS_K1_rank_0_pred_0.pkl"
)

TRUE_PKL = Path(
    r"C:\research\ms_error_geometry\results"
    r"\diffms_eg100_k1\preds"
    r"\MSG_EG100_DIFFMS_K1_rank_0_true_0.pkl"
)

MANIFEST = Path(
    r"C:\research\ms_error_geometry"
    r"\manifests\MSG_EG100_DEV_v01.csv"
)

OUT = Path(
    r"C:\research\ms_error_geometry"
    r"\results\normalized\DIFFMS_K1_EG100.csv"
)

SUMMARY = Path(
    r"C:\research\ms_error_geometry"
    r"\results\normalized\DIFFMS_K1_EG100.summary.json"
)


FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


# =============================================================================
# MOLECULE HELPERS
# =============================================================================

def fresh_copy(mol):
    """
    Binary round-trip gives a fresh RDKit object and avoids
    mutating the persisted prediction during validation.
    """
    if mol is None:
        return None

    try:
        return Chem.Mol(
            mol.ToBinary()
        )
    except Exception:
        return None


def sanitize_fresh(mol):

    m = fresh_copy(mol)

    if m is None:
        return None

    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def connectivity_key(mol):

    if mol is None:
        return None

    try:
        m = fresh_copy(mol)

        Chem.RemoveStereochemistry(m)

        key = Chem.MolToInchiKey(m)

        return (
            key[:14]
            if key
            else None
        )

    except Exception:
        return None


def canonical_connectivity_smiles(mol):

    if mol is None:
        return None

    try:
        m = fresh_copy(mol)

        Chem.RemoveStereochemistry(m)

        return Chem.MolToSmiles(
            m,
            canonical=True,
            isomericSmiles=False,
        )

    except Exception:
        return None


def formula(mol):

    if mol is None:
        return None

    try:
        return CalcMolFormula(
            fresh_copy(mol)
        )
    except Exception:
        return None


def n_components(mol):

    if mol is None:
        return None

    try:
        return len(
            Chem.GetMolFrags(
                fresh_copy(mol),
                asMols=False,
                sanitizeFrags=False,
            )
        )

    except Exception:
        return None


def tanimoto(a, b):

    if a is None or b is None:
        return None

    try:
        fa = FPGEN.GetFingerprint(
            fresh_copy(a)
        )

        fb = FPGEN.GetFingerprint(
            fresh_copy(b)
        )

        return float(
            DataStructs.TanimotoSimilarity(
                fa,
                fb,
            )
        )

    except Exception:
        return None


# =============================================================================
# LOAD
# =============================================================================

manifest = pd.read_csv(
    MANIFEST
)

eg10 = (
    manifest
    .sort_values("panel_index")
    
    .copy()
)

if eg10["connectivity_key"].nunique() != 100:
    raise RuntimeError(
        "EG100 connectivity keys are not unique."
    )

manifest_by_key = (
    eg10
    .set_index("connectivity_key")
)


with PRED_PKL.open("rb") as f:
    pred_obj = pickle.load(f)

with TRUE_PKL.open("rb") as f:
    true_obj = pickle.load(f)


print("=" * 110)
print("DIFFMS K1 EG100 STRICT EXTRACTOR")
print("=" * 110)

print(
    "Pred outer length =",
    len(pred_obj)
)

print(
    "Truth length      =",
    len(true_obj)
)


if len(pred_obj) != len(true_obj):
    raise RuntimeError(
        "Prediction/truth length mismatch."
    )

if len(true_obj) != 100:
    raise RuntimeError(
        f"Expected 100 entries, got {len(true_obj)}"
    )


# =============================================================================
# MAP EACH DIFFMS ENTRY BACK TO FROZEN MANIFEST
# =============================================================================

rows = []

seen_specs = set()


for idx, (pred_group, pickle_truth) in enumerate(
    zip(pred_obj, true_obj)
):

    # ---------------------------------------------------------
    # Identify spectrum using persisted ground truth.
    # ---------------------------------------------------------

    truth_clean = sanitize_fresh(
        pickle_truth
    )

    if truth_clean is None:
        raise RuntimeError(
            f"Persisted truth invalid at index {idx}"
        )

    pickle_truth_key = connectivity_key(
        truth_clean
    )

    if pickle_truth_key not in manifest_by_key.index:
        raise RuntimeError(
            "Persisted truth not found in EG10 manifest: "
            f"index={idx}, key={pickle_truth_key}"
        )

    meta = manifest_by_key.loc[
        pickle_truth_key
    ]

    spec = str(
        meta["spec"]
    )

    if spec in seen_specs:
        raise RuntimeError(
            f"Duplicate mapped spectrum: {spec}"
        )

    seen_specs.add(spec)


    # ---------------------------------------------------------
    # Use frozen manifest truth for evaluation.
    # ---------------------------------------------------------

    manifest_truth = Chem.MolFromSmiles(
        str(meta["smiles"])
    )

    if manifest_truth is None:
        raise RuntimeError(
            f"Manifest truth invalid: {spec}"
        )

    Chem.SanitizeMol(
        manifest_truth
    )

    manifest_truth_key = connectivity_key(
        manifest_truth
    )

    truth_identity_ok = (
        manifest_truth_key
        ==
        pickle_truth_key
    )


    # ---------------------------------------------------------
    # K=1 prediction.
    # ---------------------------------------------------------

    if not isinstance(
        pred_group,
        (list, tuple)
    ):
        raise RuntimeError(
            f"Prediction group {idx} is not list/tuple"
        )

    if len(pred_group) < 1:
        raw_pred = None
    else:
        raw_pred = pred_group[0]


    pred_clean = sanitize_fresh(
        raw_pred
    )

    rdkit_valid = (
        pred_clean is not None
    )

    components = (
        n_components(pred_clean)
        if rdkit_valid
        else None
    )

    single_component = bool(
        rdkit_valid
        and components == 1
    )

    task_valid = bool(
        rdkit_valid
        and single_component
    )


    truth_formula = formula(
        manifest_truth
    )

    pred_formula = formula(
        pred_clean
    )

    same_formula = bool(
        task_valid
        and
        pred_formula == truth_formula
    )


    pred_key = connectivity_key(
        pred_clean
    )

    exact = bool(
        task_valid
        and
        pred_key is not None
        and
        pred_key == manifest_truth_key
    )


    tani = tanimoto(
        manifest_truth,
        pred_clean,
    )


    rows.append({
        "model":
            "DiffMS-K1",

        "pickle_index":
            idx,

        "panel_index":
            int(
                meta["panel_index"]
            ),

        "spec":
            spec,

        "manifest_connectivity_key":
            pickle_truth_key,

        "pickle_truth_connectivity_key":
            pickle_truth_key,

        "truth_identity_ok":
            truth_identity_ok,

        "pred_connectivity_key":
            pred_key,

        "truth_smiles":
            canonical_connectivity_smiles(
                manifest_truth
            ),

        "pred_smiles":
            canonical_connectivity_smiles(
                pred_clean
            ),

        "truth_formula":
            truth_formula,

        "pred_formula":
            pred_formula,

        "rdkit_valid":
            rdkit_valid,

        "n_components":
            components,

        "single_component":
            single_component,

        "task_valid":
            task_valid,

        "same_formula":
            same_formula,

        "exact_connectivity":
            exact,

        "tanimoto":
            tani,
    })


out = (
    pd.DataFrame(rows)
    .sort_values("panel_index")
    .reset_index(drop=True)
)


# =============================================================================
# INTEGRITY CHECKS
# =============================================================================

if len(out) != 100:
    raise RuntimeError(
        "Output is not length 100."
    )

if out["spec"].nunique() != 100:
    raise RuntimeError(
        "Mapped spectra are not unique for EG100."
    )

if not out["truth_identity_ok"].all():
    raise RuntimeError(
        "Manifest/pickle truth mismatch."
    )

expected_specs = set(
    eg10["spec"]
)

observed_specs = set(
    out["spec"]
)

if expected_specs != observed_specs:

    print(
        "Missing =",
        sorted(
            expected_specs - observed_specs
        )
    )

    print(
        "Unexpected =",
        sorted(
            observed_specs - expected_specs
        )
    )

    raise RuntimeError(
        "EG10 spectrum-set mismatch."
    )


# =============================================================================
# WRITE
# =============================================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    OUT,
    index=False,
)


wrong = out[
    ~out["exact_connectivity"]
].copy()

valid_wrong = wrong[
    wrong["task_valid"]
].copy()


summary = {
    "model":
        "DiffMS-K1",

    "n":
        len(out),

    "manifest_truth_key_agreement":
        int(
            out[
                "truth_identity_ok"
            ].sum()
        ),

    "unique_specs":
        int(
            out["spec"].nunique()
        ),

    "rdkit_valid":
        int(
            out[
                "rdkit_valid"
            ].sum()
        ),

    "rdkit_valid_rate":
        float(
            out[
                "rdkit_valid"
            ].mean()
        ),

    "single_component":
        int(
            out[
                "single_component"
            ].sum()
        ),

    "task_valid":
        int(
            out[
                "task_valid"
            ].sum()
        ),

    "task_valid_rate":
        float(
            out[
                "task_valid"
            ].mean()
        ),

    "same_formula":
        int(
            out[
                "same_formula"
            ].sum()
        ),

    "formula_rate":
        float(
            out[
                "same_formula"
            ].mean()
        ),

    "exact":
        int(
            out[
                "exact_connectivity"
            ].sum()
        ),

    "exact_rate":
        float(
            out[
                "exact_connectivity"
            ].mean()
        ),

    "n_wrong":
        len(wrong),

    "n_task_valid_wrong":
        len(valid_wrong),

    "mean_tanimoto_task_valid":
        (
            float(
                out.loc[
                    out["task_valid"],
                    "tanimoto"
                ].mean()
            )
            if out[
                "task_valid"
            ].any()
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


SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# =============================================================================
# REPORT
# =============================================================================

print()
print(
    out[
        [
            "panel_index",
            "spec",
            "pickle_index",
            "rdkit_valid",
            "n_components",
            "task_valid",
            "same_formula",
            "exact_connectivity",
            "tanimoto",
        ]
    ].to_string(
        index=False
    )
)

print()
print("=" * 110)
print("STRICT SUMMARY")
print("=" * 110)

for k, v in summary.items():
    print(
        f"{k} = {v}"
    )

print()
print("CSV     =", OUT)
print("SUMMARY =", SUMMARY)

print()
print("DIFFMS EG100 STRICT EXTRACTION PASS")




