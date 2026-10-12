from pathlib import Path
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


CAND = Path(
    r"results\normalized\FRIGID_BASE_FM10_SMOKE10_CANDIDATES.csv"
)

TOP1 = Path(
    r"results\normalized\FRIGID_BASE_FM10_SMOKE10.csv"
)

OUT = Path(
    r"results\normalized\FRIGID_BASE_FM10_SMOKE10_STRICT_FORMULA_TOP1.csv"
)


fpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def mol(s):
    if pd.isna(s):
        return None

    try:
        m = Chem.MolFromSmiles(
            str(s),
            sanitize=False,
        )

        if m is None:
            return None

        Chem.SanitizeMol(m)
        return Chem.Mol(m)

    except Exception:
        return None


def ckey(m):
    if m is None:
        return None

    x = Chem.Mol(m)
    Chem.RemoveStereochemistry(x)

    try:
        k = Chem.MolToInchiKey(x)
        return k[:14] if k else None
    except Exception:
        return None


def tani(a, b):
    if a is None or b is None:
        return None

    return float(
        DataStructs.TanimotoSimilarity(
            fpgen.GetFingerprint(a),
            fpgen.GetFingerprint(b),
        )
    )


cand = pd.read_csv(CAND)
top1 = pd.read_csv(TOP1)

top1 = top1.set_index("spec")

rows = []

for spec, g in cand.groupby("spec"):

    g = g.sort_values(
        "saved_position"
    ).copy()

    native = top1.loc[spec]

    # --------------------------------------------------
    # Check exported position 1 against native Top-1.
    # --------------------------------------------------

    first = g.iloc[0]

    first_mol = mol(
        first["candidate_smiles"]
    )

    native_mol = mol(
        native["pred_smiles"]
    )

    exported_top1_matches_native = (
        ckey(first_mol)
        ==
        ckey(native_mol)
    )

    # --------------------------------------------------
    # First candidate in native order satisfying
    # strict molecular-formula equality.
    # --------------------------------------------------

    sf = g[
        g["same_formula"]
        & g["valid"]
    ].copy()

    if len(sf):

        selected = sf.iloc[0]

        pred = mol(
            selected["candidate_smiles"]
        )

        truth = mol(
            native["truth_smiles"]
        )

        strict_exact = (
            ckey(pred)
            ==
            ckey(truth)
        )

        strict_tani = tani(
            truth,
            pred
        )

        strict_position = int(
            selected["saved_position"]
        )

        strict_smiles = (
            selected["candidate_smiles"]
        )

    else:

        strict_exact = False
        strict_tani = None
        strict_position = None
        strict_smiles = None

    rows.append({
        "spec":
            spec,

        "exported_top1_matches_native":
            exported_top1_matches_native,

        "native_exact":
            bool(
                native[
                    "exact_connectivity"
                ]
            ),

        "native_same_formula":
            bool(
                native[
                    "same_formula"
                ]
            ),

        "native_tanimoto":
            float(
                native["tanimoto"]
            ),

        "strict_formula_available":
            len(sf) > 0,

        "strict_formula_position":
            strict_position,

        "strict_formula_exact":
            strict_exact,

        "strict_formula_tanimoto":
            strict_tani,

        "delta_strict_minus_native":
            (
                strict_tani
                - float(
                    native["tanimoto"]
                )
                if strict_tani is not None
                else None
            ),

        "strict_formula_smiles":
            strict_smiles,
    })


out = pd.DataFrame(rows)

out.to_csv(
    OUT,
    index=False
)


print("=" * 105)
print("FRIGID FM10 — NATIVE vs STRICT KNOWN-FORMULA TOP-1")
print("=" * 105)

print(
    out[
        [
            "spec",
            "exported_top1_matches_native",
            "native_same_formula",
            "strict_formula_position",
            "native_exact",
            "strict_formula_exact",
            "native_tanimoto",
            "strict_formula_tanimoto",
            "delta_strict_minus_native",
        ]
    ].to_string(index=False)
)

print()
print("=" * 105)
print("SUMMARY")
print("=" * 105)

print(
    "Exported rank-1 == native Top1 =",
    int(
        out[
            "exported_top1_matches_native"
        ].sum()
    ),
    "/",
    len(out),
)

print(
    "Strict formula candidate available =",
    int(
        out[
            "strict_formula_available"
        ].sum()
    ),
    "/",
    len(out),
)

print(
    "Native exact =",
    int(
        out[
            "native_exact"
        ].sum()
    ),
    "/",
    len(out),
)

print(
    "Strict-formula exact =",
    int(
        out[
            "strict_formula_exact"
        ].sum()
    ),
    "/",
    len(out),
)

print()
print(
    "Native mean Tanimoto =",
    out[
        "native_tanimoto"
    ].mean()
)

print(
    "Strict-formula mean Tanimoto =",
    out[
        "strict_formula_tanimoto"
    ].mean()
)

print(
    "Mean strict-native delta =",
    out[
        "delta_strict_minus_native"
    ].mean()
)

print()
print(
    "Strict candidate native rank median =",
    out[
        "strict_formula_position"
    ].median()
)

print(
    "Strict candidate native rank max =",
    out[
        "strict_formula_position"
    ].max()
)

print()
print("Written:", OUT)
