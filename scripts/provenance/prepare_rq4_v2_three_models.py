from pathlib import Path
import pandas as pd


FILES = {
    "DiffMS-K1":
        Path(r"results\normalized\DIFFMS_K1_EG100.csv"),

    "FRIGID-base@FM1":
        Path(r"results\normalized\FRIGID_BASE_FM1_EG100_HARMONIZED.csv"),

    "MS2Forge-K1":
        Path(r"results\normalized\MS2FORGE_K1_EG100.csv"),
}

SYN = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_AUDITED.parquet"
)

OUT_REAL = Path(
    r"results\RQ4_V2_REAL_THREE_MODELS.csv"
)

OUT_TRUTHS = Path(
    r"results\RQ4_V2_UNIQUE_TRUTHS.csv"
)

OUT_MISSING = Path(
    r"results\RQ4_V2_NEW_PARENTS_FOR_SYNTHETIC.csv"
)


rows = []

for model, path in FILES.items():

    d = pd.read_csv(path)

    x = d[
        d["task_valid"]
        &
        d["same_formula"]
        &
        ~d["exact_connectivity"]
    ].copy()

    print(
        model,
        "=",
        len(x)
    )

    for r in x.itertuples(index=False):

        rows.append({
            "model":
                model,

            "parent_spec":
                r.spec,

            "panel_index":
                r.panel_index,

            "truth_smiles":
                r.truth_smiles,

            "pred_smiles":
                r.pred_smiles,

            "tanimoto":
                r.tanimoto,
        })


real = pd.DataFrame(rows)

real.to_csv(
    OUT_REAL,
    index=False
)


# ----------------------------------------------------------
# One truth per parent.
# ----------------------------------------------------------

truth_rows = []

for spec, g in real.groupby(
    "parent_spec"
):

    truths = g[
        "truth_smiles"
    ].dropna().unique()

    if len(truths) != 1:
        raise RuntimeError(
            f"Truth mismatch: {spec}"
        )

    truth_rows.append({
        "sample_id":
            spec,

        "true_smiles":
            truths[0],

        "n_real_errors":
            len(g),

        "models":
            ";".join(
                sorted(
                    g["model"].unique()
                )
            ),
    })


truths = pd.DataFrame(
    truth_rows
)

truths.to_csv(
    OUT_TRUTHS,
    index=False
)


# ----------------------------------------------------------
# Compare against existing synthetic parent coverage.
# ----------------------------------------------------------

syn = pd.read_parquet(
    SYN
)

existing = set(
    syn[
        "parent_spec"
    ].astype(str)
)

needed = set(
    truths[
        "sample_id"
    ].astype(str)
)


missing = truths[
    ~truths[
        "sample_id"
    ].isin(existing)
].copy()

missing.to_csv(
    OUT_MISSING,
    index=False
)


print()
print("=" * 90)
print("RQ4 V2 THREE-MODEL REAL POOL")
print("=" * 90)

print(
    real["model"]
    .value_counts()
    .to_string()
)

print()
print(
    "REAL OBSERVATIONS =",
    len(real)
)

print(
    "UNIQUE PARENTS =",
    len(truths)
)

print(
    "EXISTING SYNTHETIC PARENTS =",
    len(existing)
)

print(
    "NEW PARENTS REQUIRING SYNTHETIC =",
    len(missing)
)

print()

if len(missing):
    print(
        missing[
            [
                "sample_id",
                "n_real_errors",
                "models",
            ]
        ].to_string(
            index=False
        )
    )

print()
print("REAL    =", OUT_REAL)
print("TRUTHS  =", OUT_TRUTHS)
print("MISSING =", OUT_MISSING)
