from pathlib import Path
import pandas as pd

FILES = {
    "DiffMS-K1": Path(
        r"results\normalized\DIFFMS_K1_EG358_CONFIRM.csv"
    ),
    "FRIGID-base@FM1": Path(
        r"results\normalized\FRIGID_BASE_FM1_EG358_CONFIRM_HARMONIZED.csv"
    ),
    "MS2Forge-K1": Path(
        r"results\normalized\MS2FORGE_K1_EG358_CONFIRM.csv"
    ),
}

OUT_REAL = Path(
    r"results\RQ4_CONFIRM_REAL_THREE_MODELS_EG358.csv"
)

OUT_TRUTHS = Path(
    r"results\RQ4_CONFIRM_UNIQUE_TRUTHS_EG358.csv"
)

rows = []

for model, path in FILES.items():

    d = pd.read_csv(path)

    q = d[
        d["task_valid"].astype(bool)
        &
        d["same_formula"].astype(bool)
        &
        ~d["exact_connectivity"].astype(bool)
    ].copy()

    print(
        model,
        "=",
        len(q)
    )

    q["model"] = model

    rows.append(
        q[
            [
                "model",
                "spec",
                "truth_smiles",
                "pred_smiles",
                "tanimoto",
            ]
        ]
    )


real = pd.concat(
    rows,
    ignore_index=True
)


print()
print("=" * 90)
print("RQ4 EG358 CONFIRMATORY REAL POOL")
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
print("ERRORS PER PARENT")

counts = (
    real.groupby("spec")
    .size()
    .value_counts()
    .sort_index()
)

for n_errors, n_parents in counts.items():
    print(
        f"{n_errors} real error(s): "
        f"{n_parents} parents"
    )


# Hard expected observation count.
assert len(real) == 430, (
    f"Expected 430 real errors, got {len(real)}"
)


real.to_csv(
    OUT_REAL,
    index=False
)


# One unique truth molecule for every spectrum/parent.
truth_check = (
    real.groupby("spec")["truth_smiles"]
    .nunique()
)

bad = truth_check[
    truth_check != 1
]

if len(bad):
    raise RuntimeError(
        f"Truth mismatch for {len(bad)} parent(s)"
    )


truths = (
    real[
        [
            "spec",
            "truth_smiles",
        ]
    ]
    .drop_duplicates(subset=["spec"])
    .rename(
        columns={
            "spec": "sample_id",
            "truth_smiles": "true_smiles",
        }
    )
    .sort_values("sample_id")
    .reset_index(drop=True)
)


assert (
    truths["sample_id"].nunique()
    ==
    len(truths)
)


truths.to_csv(
    OUT_TRUTHS,
    index=False
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
    "RQ4 CONFIRM REAL POOL PASS"
)
