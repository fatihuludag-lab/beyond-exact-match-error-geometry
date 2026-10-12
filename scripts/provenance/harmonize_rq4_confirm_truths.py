from pathlib import Path

import pandas as pd
from rdkit import Chem


ROOT = Path(r"C:\research\ms_error_geometry")

MANIFEST = (
    ROOT / "manifests" /
    "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv"
)

FILES = {
    "DiffMS-K1":
        ROOT / "results" / "normalized" /
        "DIFFMS_K1_EG358_CONFIRM.csv",

    "FRIGID-base@FM1":
        ROOT / "results" / "normalized" /
        "FRIGID_BASE_FM1_EG358_CONFIRM_HARMONIZED.csv",

    "MS2Forge-K1":
        ROOT / "results" / "normalized" /
        "MS2FORGE_K1_EG358_CONFIRM.csv",
}

OUT_REAL = (
    ROOT / "results" /
    "RQ4_CONFIRM_REAL_THREE_MODELS_EG358.csv"
)

OUT_TRUTHS = (
    ROOT / "results" /
    "RQ4_CONFIRM_UNIQUE_TRUTHS_EG358.csv"
)


def to_bool(s):
    if pd.api.types.is_bool_dtype(s):
        return s

    return (
        s.astype(str)
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
        })
    )


def fresh_mol(smiles):
    if pd.isna(smiles):
        return None

    try:
        m = Chem.MolFromSmiles(
            str(smiles),
            sanitize=False,
        )

        if m is None:
            return None

        Chem.SanitizeMol(m)

        return m

    except Exception:
        return None


def canonical_no_stereo(smiles):
    m = fresh_mol(smiles)

    if m is None:
        raise RuntimeError(
            f"Invalid SMILES: {smiles}"
        )

    Chem.RemoveStereochemistry(m)

    return Chem.MolToSmiles(
        m,
        canonical=True,
        isomericSmiles=False,
    )


def connectivity_key_from_smiles(smiles):
    m = fresh_mol(smiles)

    if m is None:
        return None

    try:
        Chem.RemoveStereochemistry(m)

        key = Chem.MolToInchiKey(m)

        return key[:14] if key else None

    except Exception:
        return None


# ============================================================
# Frozen manifest = single authoritative truth source
# ============================================================

manifest = pd.read_csv(MANIFEST)

assert len(manifest) == 358
assert manifest["spec"].nunique() == 358

truth_candidates = [
    "canonical_smiles",
    "smiles",
    "truth_smiles",
]

truth_col = next(
    (
        c
        for c in truth_candidates
        if c in manifest.columns
    ),
    None,
)

if truth_col is None:
    raise RuntimeError(
        "No truth SMILES column found in manifest. "
        f"Columns={manifest.columns.tolist()}"
    )

if "connectivity_key" not in manifest.columns:
    raise RuntimeError(
        "Frozen manifest has no connectivity_key."
    )


manifest["spec"] = manifest["spec"].astype(str)

manifest["manifest_connectivity_key"] = (
    manifest["connectivity_key"]
    .astype(str)
    .str[:14]
)

manifest["manifest_truth_smiles"] = (
    manifest[truth_col]
    .apply(canonical_no_stereo)
)

manifest["computed_truth_key"] = (
    manifest["manifest_truth_smiles"]
    .apply(connectivity_key_from_smiles)
)


bad_manifest = manifest[
    manifest["computed_truth_key"]
    !=
    manifest["manifest_connectivity_key"]
]

if len(bad_manifest):
    print(
        bad_manifest[
            [
                "spec",
                truth_col,
                "manifest_connectivity_key",
                "computed_truth_key",
            ]
        ].to_string(index=False)
    )

    raise RuntimeError(
        f"Manifest connectivity audit failed for "
        f"{len(bad_manifest)} spectra."
    )


manifest_lookup = (
    manifest.set_index("spec")
)


# ============================================================
# Extract same-formula wrong observations
# ============================================================

all_rows = []

raw_string_mismatch = 0
connectivity_mismatch = 0


for model, path in FILES.items():

    d = pd.read_csv(path)

    d["task_valid"] = to_bool(
        d["task_valid"]
    )

    d["same_formula"] = to_bool(
        d["same_formula"]
    )

    d["exact_connectivity"] = to_bool(
        d["exact_connectivity"]
    )

    q = d[
        d["task_valid"]
        &
        d["same_formula"]
        &
        ~d["exact_connectivity"]
    ].copy()

    print(
        model,
        "=",
        len(q)
    )

    for r in q.itertuples(index=False):

        spec = str(r.spec)

        if spec not in manifest_lookup.index:
            raise RuntimeError(
                f"{model}: unknown spec {spec}"
            )

        meta = manifest_lookup.loc[
            spec
        ]

        authoritative_truth = (
            meta["manifest_truth_smiles"]
        )

        authoritative_key = (
            meta["manifest_connectivity_key"]
        )

        model_truth = str(
            r.truth_smiles
        )

        # Raw/canonical string mismatch is not necessarily
        # a molecular mismatch.
        try:
            model_truth_canonical = (
                canonical_no_stereo(
                    model_truth
                )
            )
        except Exception:
            model_truth_canonical = None

        if (
            model_truth_canonical
            != authoritative_truth
        ):
            raw_string_mismatch += 1

        model_truth_key = (
            connectivity_key_from_smiles(
                model_truth
            )
        )

        if model_truth_key != authoritative_key:
            connectivity_mismatch += 1

            print(
                "CONNECTIVITY MISMATCH:",
                model,
                spec,
                model_truth_key,
                authoritative_key,
            )

        all_rows.append({
            "model":
                model,

            "spec":
                spec,

            "panel_index":
                int(meta["panel_index"]),

            "common_support_index":
                int(
                    meta[
                        "common_support_index"
                    ]
                ),

            "truth_smiles":
                authoritative_truth,

            "truth_connectivity_key":
                authoritative_key,

            "pred_smiles":
                r.pred_smiles,

            "tanimoto":
                float(r.tanimoto),
        })


real = pd.DataFrame(
    all_rows
)


# ============================================================
# Hard gates
# ============================================================

print()
print("=" * 90)
print("TRUTH HARMONIZATION AUDIT")
print("=" * 90)

print(
    "Representation-level mismatches =",
    raw_string_mismatch
)

print(
    "Connectivity mismatches =",
    connectivity_mismatch
)


if connectivity_mismatch != 0:
    raise RuntimeError(
        "At least one model truth does not match "
        "the frozen manifest at connectivity level."
    )


assert len(real) == 430

assert (
    real["spec"].nunique()
    == 333
)


# Every parent must now have exactly one authoritative truth.
truth_count = (
    real.groupby("spec")[
        "truth_connectivity_key"
    ]
    .nunique()
)

assert (
    truth_count == 1
).all()


print()
print("=" * 90)
print("RQ4 EG358 CONFIRMATORY REAL POOL — HARMONIZED")
print("=" * 90)

print(
    "REAL OBSERVATIONS =",
    len(real)
)

print(
    "UNIQUE PARENTS =",
    real["spec"].nunique()
)

print()
print("MODEL COUNTS")

print(
    real["model"]
    .value_counts()
    .to_string()
)

print()
print("ERRORS PER PARENT")

parent_counts = (
    real.groupby("spec")
    .size()
    .value_counts()
    .sort_index()
)

for n_errors, n_parents in parent_counts.items():

    print(
        f"{n_errors} real error(s): "
        f"{n_parents} parents"
    )


# ============================================================
# Save corrected real pool
# ============================================================

real = (
    real.sort_values(
        [
            "common_support_index",
            "model",
        ]
    )
    .reset_index(drop=True)
)

real.to_csv(
    OUT_REAL,
    index=False,
)


# ============================================================
# One frozen truth per parent for synthetic corruption
# ============================================================

truths = (
    real[
        [
            "spec",
            "truth_smiles",
            "truth_connectivity_key",
            "panel_index",
            "common_support_index",
        ]
    ]
    .drop_duplicates(
        subset=["spec"]
    )
    .rename(
        columns={
            "spec":
                "sample_id",

            "truth_smiles":
                "true_smiles",
        }
    )
    .sort_values(
        "common_support_index"
    )
    .reset_index(drop=True)
)


assert len(truths) == 333
assert truths["sample_id"].nunique() == 333
assert truths["truth_connectivity_key"].nunique() == 333


truths.to_csv(
    OUT_TRUTHS,
    index=False,
)


print()
print(
    "REAL =",
    OUT_REAL
)

print(
    "TRUTHS =",
    OUT_TRUTHS
)

print()
print(
    "RQ4 CONFIRM TRUTH HARMONIZATION PASS"
)
