from pathlib import Path
import json

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


CACHE = Path(
    r"results\ms2forge_eg358_confirm_k1"
    r"\candidate_cache_test_n1_T20_r1.0_seed2026_zms_real_bs2.jsonl"
)

MANIFEST = Path(
    r"manifests\MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv"
)

OUT = Path(
    r"results\normalized\MS2FORGE_K1_EG358_CONFIRM.csv"
)

SUMMARY = Path(
    r"results\normalized\MS2FORGE_K1_EG358_CONFIRM.summary.json"
)

FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def mol(s):

    if s is None or pd.isna(s):
        return None

    try:
        m = Chem.MolFromSmiles(
            str(s),
            sanitize=False
        )

        if m is None:
            return None

        Chem.SanitizeMol(m)

        return Chem.Mol(m)

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
                sanitizeFrags=False
            )
        )
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


def formula(m):

    if m is None:
        return None

    try:
        return CalcMolFormula(
            Chem.Mol(m)
        )
    except Exception:
        return None


def can_smiles(m):

    if m is None:
        return None

    x = Chem.Mol(m)
    Chem.RemoveStereochemistry(x)

    return Chem.MolToSmiles(
        x,
        canonical=True,
        isomericSmiles=False
    )


def tani(a, b):

    if a is None or b is None:
        return None

    try:
        return float(
            DataStructs.TanimotoSimilarity(
                FPGEN.GetFingerprint(a),
                FPGEN.GetFingerprint(b)
            )
        )
    except Exception:
        return None


manifest = (
    pd.read_csv(MANIFEST)
    .sort_values("panel_index")
)

manifest_by_spec = (
    manifest.set_index("spec")
)


records = []

with CACHE.open(
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        r = json.loads(line)

        if r.get("type") != "mol_candidates":
            continue

        records.append(r)


if len(records) != 358:
    raise RuntimeError(
        f"Expected 358 mol records, got {len(records)}"
    )


rows = []

for r in records:

    spec = str(
        r["spec_id"]
    )

    if spec not in manifest_by_spec.index:
        raise RuntimeError(
            f"Unknown spec_id: {spec}"
        )

    meta = manifest_by_spec.loc[spec]

    truth = mol(
        meta["smiles"]
    )

    if truth is None:
        raise RuntimeError(
            f"Invalid truth: {spec}"
        )

    pred_stats = r["pred"]

    # Native diffms_2d ranking:
    # frequency-ranked valid + connected non-isomeric candidates.
    connected = (
        pred_stats.get(
            "connected_smiles_nonisomeric_counts",
            []
        )
    )

    valid_all = (
        pred_stats.get(
            "valid_smiles_counts",
            []
        )
    )

    native_candidate = (
        connected[0][0]
        if connected
        else None
    )

    # If disconnected but sanitizable, retain a representation
    # for diagnostic validity, but it is not task-valid.
    diagnostic_candidate = (
        native_candidate
        if native_candidate is not None
        else (
            valid_all[0][0]
            if valid_all
            else None
        )
    )

    pred = mol(
        diagnostic_candidate
    )

    rdkit_valid = (
        pred is not None
    )

    components = (
        n_components(pred)
        if pred is not None
        else None
    )

    task_valid = bool(
        native_candidate is not None
        and pred is not None
        and components == 1
    )

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

    pred_key = conn_key(
        pred
    )

    exact = bool(
        task_valid
        and pred_key is not None
        and pred_key == truth_key
    )

    rows.append({
        "model":
            "MS2Forge-K1",

        "panel_index":
            int(meta["panel_index"]),

        "spec":
            spec,

        "truth_smiles":
            can_smiles(truth),

        "pred_smiles":
            can_smiles(pred),

        "truth_formula":
            truth_formula,

        "pred_formula":
            pred_formula,

        "rdkit_valid":
            rdkit_valid,

        "n_components":
            components,

        "single_component":
            bool(
                rdkit_valid
                and components == 1
            ),

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

        "raw_n_generated":
            int(
                pred_stats.get(
                    "n_generated",
                    0
                )
            ),

        "raw_n_valid":
            int(
                pred_stats.get(
                    "n_valid",
                    0
                )
            ),

        "raw_n_valid_connected":
            int(
                pred_stats.get(
                    "n_valid_connected",
                    0
                )
            ),
    })


out = (
    pd.DataFrame(rows)
    .sort_values("panel_index")
    .reset_index(drop=True)
)


if out["spec"].nunique() != 358:
    raise RuntimeError(
        "Spectrum mapping is not unique."
    )


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    OUT,
    index=False
)


wrong = out[
    ~out["exact_connectivity"]
]

valid_wrong = wrong[
    wrong["task_valid"]
]


summary = {
    "model":
        "MS2Forge-K1",

    "n":
        len(out),

    "rdkit_valid":
        int(out["rdkit_valid"].sum()),

    "task_valid":
        int(out["task_valid"].sum()),

    "same_formula":
        int(out["same_formula"].sum()),

    "exact":
        int(out["exact_connectivity"].sum()),

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


SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


print("=" * 100)
print("MS2FORGE K1 EG358 CONFIRM STRICT NORMALIZATION")
print("=" * 100)

print(
    out[
        [
            "panel_index",
            "spec",
            "rdkit_valid",
            "n_components",
            "task_valid",
            "same_formula",
            "exact_connectivity",
            "tanimoto",
        ]
    ].to_string(index=False)
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

for k, v in summary.items():
    print(f"{k} = {v}")

print()
print("CSV =", OUT)
print("SUMMARY =", SUMMARY)


