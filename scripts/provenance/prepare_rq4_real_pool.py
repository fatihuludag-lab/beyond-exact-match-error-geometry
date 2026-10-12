from pathlib import Path
import pandas as pd

D = Path(
    r"results\normalized\DIFFMS_K1_EG100.csv"
)

F = Path(
    r"results\normalized\FRIGID_BASE_FM1_EG100_HARMONIZED.csv"
)

OUT_REAL = Path(
    r"results\RQ4_REAL_SAME_FORMULA_WRONG_EG100.csv"
)

OUT_UNIQUE = Path(
    r"results\RQ4_REAL_UNIQUE_TRUTHS_EG100.csv"
)

d = pd.read_csv(D)
f = pd.read_csv(F)


def extract(df, model):

    x = df[
        df["task_valid"]
        &
        df["same_formula"]
        &
        ~df["exact_connectivity"]
    ].copy()

    x["model"] = model

    return x[
        [
            "model",
            "panel_index",
            "spec",
            "truth_smiles",
            "pred_smiles",
            "tanimoto",
        ]
    ]


real = pd.concat(
    [
        extract(d, "DiffMS-K1"),
        extract(f, "FRIGID-base@FM1"),
    ],
    ignore_index=True,
)

real.to_csv(
    OUT_REAL,
    index=False
)


# One truth molecule per spectrum.
rows = []

for spec, g in real.groupby("spec"):

    truths = g["truth_smiles"].dropna().unique()

    if len(truths) != 1:
        raise RuntimeError(
            f"Truth mismatch for {spec}: {truths}"
        )

    rows.append({
        "sample_id":
            spec,

        "true_smiles":
            truths[0],

        "n_real_generator_errors":
            len(g),

        "models_wrong_same_formula":
            ";".join(
                sorted(
                    g["model"].unique()
                )
            ),
    })


unique = (
    pd.DataFrame(rows)
    .sort_values("sample_id")
    .reset_index(drop=True)
)

unique.to_csv(
    OUT_UNIQUE,
    index=False
)


print("=" * 100)
print("RQ4 REAL ERROR POOL")
print("=" * 100)

print(
    real["model"]
    .value_counts()
    .to_string()
)

print()
print(
    "REAL ERROR OBSERVATIONS =",
    len(real)
)

print(
    "UNIQUE TRUTH MOLECULES =",
    real["spec"].nunique()
)

print()
print(
    unique[
        "n_real_generator_errors"
    ].value_counts()
    .sort_index()
    .to_string()
)

print()
print("REAL   =", OUT_REAL)
print("UNIQUE =", OUT_UNIQUE)
