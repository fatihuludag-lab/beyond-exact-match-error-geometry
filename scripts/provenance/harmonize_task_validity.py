from pathlib import Path
import argparse
import pandas as pd
from rdkit import Chem


def n_components(smiles):
    if pd.isna(smiles):
        return None

    try:
        m = Chem.MolFromSmiles(
            str(smiles),
            sanitize=False
        )

        if m is None:
            return None

        Chem.SanitizeMol(m)

        return len(
            Chem.GetMolFrags(
                m,
                asMols=False,
                sanitizeFrags=False
            )
        )

    except Exception:
        return None


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    required=True
)

parser.add_argument(
    "--output",
    required=True
)

args = parser.parse_args()

df = pd.read_csv(args.input)

df["n_components"] = (
    df["pred_smiles"]
    .map(n_components)
)

df["single_component"] = (
    df["valid"]
    &
    df["n_components"].eq(1)
)

df["task_valid"] = (
    df["valid"]
    &
    df["single_component"]
)

# Preserve original formula flag for auditing.
df["raw_same_formula"] = (
    df["same_formula"]
)

# Primary benchmark definition.
df["same_formula"] = (
    df["task_valid"]
    &
    df["raw_same_formula"]
)

# Exact must also satisfy task validity.
df["exact_connectivity"] = (
    df["task_valid"]
    &
    df["exact_connectivity"]
)

out = Path(args.output)
out.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    out,
    index=False
)

print("=" * 90)
print("FRIGID TASK-VALIDITY HARMONIZATION")
print("=" * 90)

print("n =", len(df))
print("RDKit valid =", int(df["valid"].sum()))
print(
    "single component =",
    int(df["single_component"].sum())
)
print(
    "task valid =",
    int(df["task_valid"].sum())
)
print(
    "same formula =",
    int(df["same_formula"].sum())
)
print(
    "exact =",
    int(df["exact_connectivity"].sum())
)

print()
print("Written:", out)
