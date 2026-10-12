from pathlib import Path
import argparse
import json

import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def strict_mol(smiles):
    if pd.isna(smiles):
        return None

    text = str(smiles).strip()

    if not text:
        return None

    try:
        mol = Chem.MolFromSmiles(
            text,
            sanitize=False,
        )

        if mol is None:
            return None

        Chem.SanitizeMol(mol)

        # Fresh independent copy.
        mol = Chem.Mol(mol)

        return mol

    except Exception:
        return None


def connectivity_smiles(mol):
    if mol is None:
        return None

    m = Chem.Mol(mol)

    Chem.RemoveStereochemistry(m)

    return Chem.MolToSmiles(
        m,
        canonical=True,
        isomericSmiles=False,
    )


def connectivity_key(mol):
    if mol is None:
        return None

    try:
        m = Chem.Mol(mol)

        Chem.RemoveStereochemistry(m)

        ik = Chem.MolToInchiKey(m)

        if not ik:
            return None

        return ik[:14]

    except Exception:
        return None


def formula(mol):
    if mol is None:
        return None

    try:
        return CalcMolFormula(
            Chem.Mol(mol)
        )
    except Exception:
        return None


def tanimoto(a, b):
    if a is None or b is None:
        return None

    fa = FPGEN.GetFingerprint(
        Chem.Mol(a)
    )

    fb = FPGEN.GetFingerprint(
        Chem.Mol(b)
    )

    return float(
        DataStructs.TanimotoSimilarity(
            fa,
            fb,
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest
    )

    raw = pd.read_csv(
        args.input
    )

    # ------------------------------------------------------------
    # FRIGID detailed_results.csv schema
    # ------------------------------------------------------------

    if "spec_name" not in raw.columns:
        raise RuntimeError(
            "Expected FRIGID column 'spec_name'. "
            f"Columns={list(raw.columns)}"
        )

    if "proposal_smiles" not in raw.columns:
        raise RuntimeError(
            "Expected FRIGID column "
            "'proposal_smiles'."
        )

    if "target_smiles" not in raw.columns:
        raise RuntimeError(
            "Expected FRIGID column "
            "'target_smiles'."
        )

    manifest_lookup = (
        manifest
        .set_index("spec")
    )

    rows = []

    for r in raw.itertuples(index=False):

        spec = str(r.spec_name)

        if spec not in manifest_lookup.index:
            raise RuntimeError(
                f"{spec} not found in manifest"
            )

        meta = manifest_lookup.loc[spec]

        true_raw = str(
            r.target_smiles
        )

        pred_raw = (
            None
            if pd.isna(r.proposal_smiles)
            else str(r.proposal_smiles)
        )

        true_mol = strict_mol(
            true_raw
        )

        pred_mol = strict_mol(
            pred_raw
        )

        if true_mol is None:
            raise RuntimeError(
                f"Invalid truth: {spec}"
            )

        true_can = connectivity_smiles(
            true_mol
        )

        pred_can = connectivity_smiles(
            pred_mol
        )

        true_formula = formula(
            true_mol
        )

        pred_formula = formula(
            pred_mol
        )

        true_ckey = connectivity_key(
            true_mol
        )

        pred_ckey = connectivity_key(
            pred_mol
        )

        valid = (
            pred_mol is not None
        )

        same_formula = bool(
            valid
            and
            pred_formula == true_formula
        )

        exact_connectivity = bool(
            valid
            and
            pred_ckey is not None
            and
            pred_ckey == true_ckey
        )

        tani = tanimoto(
            true_mol,
            pred_mol,
        )

        rows.append(
            {
                "model":
                    args.model,

                "panel_index":
                    int(
                        meta["panel_index"]
                    ),

                "spec":
                    spec,

                "manifest_connectivity_key":
                    meta[
                        "connectivity_key"
                    ],

                "truth_connectivity_key":
                    true_ckey,

                "pred_connectivity_key":
                    pred_ckey,

                "truth_smiles":
                    true_can,

                "pred_smiles":
                    pred_can,

                "truth_formula":
                    true_formula,

                "pred_formula":
                    pred_formula,

                "valid":
                    valid,

                "same_formula":
                    same_formula,

                "exact_connectivity":
                    exact_connectivity,

                "tanimoto":
                    tani,

                "raw_total_generated":
                    getattr(
                        r,
                        "total_generated",
                        None,
                    ),

                "raw_total_valid":
                    getattr(
                        r,
                        "total_valid",
                        None,
                    ),

                "raw_total_formula_matched":
                    getattr(
                        r,
                        "total_formula_matched",
                        None,
                    ),

                "raw_mist_tanimoto":
                    getattr(
                        r,
                        "mist_tanimoto",
                        None,
                    ),

                "raw_reported_tanimoto":
                    getattr(
                        r,
                        "tanimoto_top1",
                        None,
                    ),
            }
        )

    out = (
        pd.DataFrame(rows)
        .sort_values(
            "panel_index"
        )
        .reset_index(drop=True)
    )

    out_path = Path(
        args.output
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        out_path,
        index=False,
    )

    n = len(out)

    n_valid = int(
        out["valid"].sum()
    )

    n_formula = int(
        out["same_formula"].sum()
    )

    n_exact = int(
        out["exact_connectivity"].sum()
    )

    wrong = out[
        ~out["exact_connectivity"]
    ]

    valid_wrong = wrong[
        wrong["valid"]
    ]

    formula_wrong = wrong[
        wrong["same_formula"]
    ]

    summary = {
        "model":
            args.model,

        "n":
            n,

        "valid":
            n_valid,

        "valid_rate":
            n_valid / n
            if n else None,

        "same_formula":
            n_formula,

        "formula_rate":
            n_formula / n
            if n else None,

        "exact":
            n_exact,

        "exact_rate":
            n_exact / n
            if n else None,

        "n_wrong":
            len(wrong),

        "n_valid_wrong":
            len(valid_wrong),

        "n_formula_matched_wrong":
            len(formula_wrong),

        "mean_tanimoto_all_valid":
            float(
                out.loc[
                    out["valid"],
                    "tanimoto",
                ].mean()
            )
            if n_valid
            else None,

        "mean_tanimoto_wrong_valid":
            float(
                valid_wrong[
                    "tanimoto"
                ].mean()
            )
            if len(valid_wrong)
            else None,

        "median_tanimoto_wrong_valid":
            float(
                valid_wrong[
                    "tanimoto"
                ].median()
            )
            if len(valid_wrong)
            else None,

        "manifest_truth_key_agreement":
            int(
                (
                    out[
                        "manifest_connectivity_key"
                    ]
                    ==
                    out[
                        "truth_connectivity_key"
                    ]
                ).sum()
            ),
    }

    summary_path = (
        out_path.with_suffix(
            ".summary.json"
        )
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("COMMON ERROR-GEOMETRY NORMALIZER")
    print("=" * 100)

    print(
        out[
            [
                "panel_index",
                "spec",
                "valid",
                "same_formula",
                "exact_connectivity",
                "tanimoto",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for k, v in summary.items():
        print(
            f"{k} = {v}"
        )

    print()
    print("CSV     =", out_path)
    print("SUMMARY =", summary_path)


if __name__ == "__main__":
    main()
