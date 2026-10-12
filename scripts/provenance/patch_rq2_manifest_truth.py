from pathlib import Path
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


ROOT = Path(r"C:\research\ms_error_geometry")

OLD_MAP = (
    ROOT / "results" / "mces" /
    "EG358_CONFIRM_RQ2_MCES_MAP.csv"
)

OLD_OUT = (
    ROOT / "results" / "mces" /
    "EG358_CONFIRM_RQ2_MCES_OUT.csv"
)

FIX_MAP = (
    ROOT / "results" / "mces" /
    "RQ2_CONFIRM_TRUTH_FIX_MAP.csv"
)

FIX_OUT = (
    ROOT / "results" / "mces" /
    "RQ2_CONFIRM_TRUTH_FIX_OUT.csv"
)

MANIFEST = (
    ROOT / "manifests" /
    "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv"
)

NEW_MAP = (
    ROOT / "results" / "mces" /
    "EG358_CONFIRM_RQ2_MCES_MAP_MANIFEST_TRUTH.csv"
)

NEW_OUT = (
    ROOT / "results" / "mces" /
    "EG358_CONFIRM_RQ2_MCES_OUT_MANIFEST_TRUTH.csv"
)


FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def mol(s):
    m = Chem.MolFromSmiles(
        str(s),
        sanitize=False
    )

    if m is None:
        raise RuntimeError(
            f"Invalid SMILES: {s}"
        )

    Chem.SanitizeMol(m)
    Chem.RemoveStereochemistry(m)

    return m


def canonical(s):
    return Chem.MolToSmiles(
        mol(s),
        canonical=True,
        isomericSmiles=False
    )


def tani(a, b):
    ma = mol(a)
    mb = mol(b)

    return float(
        DataStructs.TanimotoSimilarity(
            FPGEN.GetFingerprint(ma),
            FPGEN.GetFingerprint(mb)
        )
    )


def formula(s):
    return CalcMolFormula(
        mol(s)
    )


# ============================================================
# Original 585 exact results
# ============================================================

old_map = pd.read_csv(OLD_MAP)

old_out = pd.read_csv(
    OLD_OUT,
    header=None,
    names=[
        "pair_id",
        "mces",
        "time_sec",
        "mode",
    ]
)

assert len(old_map) == 585
assert len(old_out) == 585
assert set(old_out["mode"]) == {1}


base = old_map.merge(
    old_out,
    on="pair_id",
    validate="one_to_one"
)


# ============================================================
# 274 corrections
# ============================================================

fix_map = pd.read_csv(FIX_MAP)

fix_out = pd.read_csv(
    FIX_OUT,
    header=None,
    names=[
        "fix_pair_id",
        "fixed_mces",
        "fixed_time_sec",
        "fixed_mode",
    ]
)

assert len(fix_map) == 274
assert len(fix_out) == 274
assert set(fix_out["fixed_mode"]) == {1}


fix = fix_map.merge(
    fix_out,
    on="fix_pair_id",
    validate="one_to_one"
)

fix_lookup = (
    fix.set_index("pair_id")
)


# ============================================================
# Manifest authoritative truth
# ============================================================

manifest = pd.read_csv(MANIFEST)

truth_col = next(
    c
    for c in [
        "canonical_smiles",
        "smiles",
        "truth_smiles",
    ]
    if c in manifest.columns
)

manifest["spec"] = (
    manifest["spec"]
    .astype(str)
)

manifest["manifest_truth"] = (
    manifest[truth_col]
    .apply(canonical)
)

manifest_lookup = (
    manifest.set_index("spec")
)


# ============================================================
# Patch MCES + harmonize truth-dependent endpoints
# ============================================================

patched = []

n_replaced = 0
n_changed_mces = 0
max_abs_change = 0.0


for r in base.itertuples(index=False):

    pair_id = int(r.pair_id)
    spec = str(r.spec)

    truth = manifest_lookup.loc[
        spec,
        "manifest_truth"
    ]

    pred = canonical(
        r.pred_smiles
    )

    old_mces = float(r.mces)

    if pair_id in fix_lookup.index:

        f = fix_lookup.loc[
            pair_id
        ]

        new_mces = float(
            f["fixed_mces"]
        )

        time_sec = float(
            f["fixed_time_sec"]
        )

        mode = int(
            f["fixed_mode"]
        )

        n_replaced += 1

        diff = abs(
            new_mces - old_mces
        )

        if diff > 0:
            n_changed_mces += 1

        max_abs_change = max(
            max_abs_change,
            diff
        )

    else:

        new_mces = old_mces
        time_sec = float(
            r.time_sec
        )
        mode = int(
            r.mode
        )


    patched.append({
        "pair_id":
            pair_id,

        "spec":
            spec,

        "model":
            r.model,

        "same_formula":
            bool(
                formula(truth)
                == formula(pred)
            ),

        "tanimoto":
            tani(
                truth,
                pred
            ),

        "truth_smiles":
            truth,

        "pred_smiles":
            pred,

        "mces":
            new_mces,

        "time_sec":
            time_sec,

        "mode":
            mode,
    })


out = (
    pd.DataFrame(patched)
    .sort_values("pair_id")
    .reset_index(drop=True)
)


assert len(out) == 585
assert out["pair_id"].nunique() == 585
assert set(out["mode"]) == {1}


print("=" * 100)
print("RQ2 AUTHORITATIVE MANIFEST-TRUTH PATCH")
print("=" * 100)

print(
    "TOTAL =",
    len(out)
)

print(
    "MCES RECOMPUTED =",
    n_replaced
)

print(
    "MCES VALUES ACTUALLY CHANGED =",
    n_changed_mces
)

print(
    "MAX ABS MCES CHANGE =",
    max_abs_change
)


# map schema
out[
    [
        "pair_id",
        "spec",
        "model",
        "same_formula",
        "tanimoto",
        "truth_smiles",
        "pred_smiles",
    ]
].to_csv(
    NEW_MAP,
    index=False
)


# exact MCES output schema
out[
    [
        "pair_id",
        "mces",
        "time_sec",
        "mode",
    ]
].to_csv(
    NEW_OUT,
    index=False,
    header=False
)


print()
print(
    "NEW MAP =",
    NEW_MAP
)

print(
    "NEW OUT =",
    NEW_OUT
)

print()
print(
    "RQ2 MANIFEST-TRUTH PATCH PASS"
)
