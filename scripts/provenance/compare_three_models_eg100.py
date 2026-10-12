from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


FILES = {
    "DiffMS":
        Path(r"results\normalized\DIFFMS_K1_EG100.csv"),

    "FRIGID":
        Path(r"results\normalized\FRIGID_BASE_FM1_EG100_HARMONIZED.csv"),

    "MS2Forge":
        Path(r"results\normalized\MS2FORGE_K1_EG100.csv"),
}

OUT = Path(
    r"results\THREE_MODEL_PAIRED_EG100.csv"
)

SEED = 20261008
N_BOOT = 30000


def load_model(name, path):

    d = pd.read_csv(path)

    keep = d[
        [
            "panel_index",
            "spec",
            "task_valid",
            "same_formula",
            "exact_connectivity",
            "tanimoto",
        ]
    ].copy()

    return keep.rename(
        columns={
            "task_valid":
                f"{name}_valid",

            "same_formula":
                f"{name}_formula",

            "exact_connectivity":
                f"{name}_exact",

            "tanimoto":
                f"{name}_tanimoto",
        }
    )


def sign_test(delta):

    x = np.asarray(delta, dtype=float)

    pos = int((x > 0).sum())
    neg = int((x < 0).sum())
    ties = int((x == 0).sum())

    n = pos + neg

    if n == 0:
        return pos, neg, ties, 1.0

    lo = min(pos, neg)

    tail = sum(
        comb(n, i)
        for i in range(lo + 1)
    ) / (2 ** n)

    return (
        pos,
        neg,
        ties,
        min(1.0, 2.0 * tail),
    )


def bootstrap(delta):

    x = np.asarray(
        delta,
        dtype=float
    )

    rng = np.random.default_rng(
        SEED
    )

    idx = rng.integers(
        0,
        len(x),
        size=(
            N_BOOT,
            len(x)
        )
    )

    z = x[idx]

    return {
        "mean":
            float(x.mean()),

        "mean_low":
            float(
                np.quantile(
                    z.mean(axis=1),
                    0.025
                )
            ),

        "mean_high":
            float(
                np.quantile(
                    z.mean(axis=1),
                    0.975
                )
            ),

        "median":
            float(
                np.median(x)
            ),
    }


# ------------------------------------------------------------
# Merge all three models
# ------------------------------------------------------------

models = {
    name: load_model(name, path)
    for name, path in FILES.items()
}

x = models["DiffMS"]

for name in [
    "FRIGID",
    "MS2Forge",
]:
    x = x.merge(
        models[name],
        on=[
            "panel_index",
            "spec",
        ],
        validate="one_to_one",
    )

assert len(x) == 100


# ------------------------------------------------------------
# Global model summary
# ------------------------------------------------------------

print("=" * 105)
print("THREE-MODEL EG100 SUMMARY")
print("=" * 105)

for name in FILES:

    print()
    print(name)

    print(
        "valid =",
        int(
            x[
                f"{name}_valid"
            ].sum()
        )
    )

    print(
        "formula =",
        int(
            x[
                f"{name}_formula"
            ].sum()
        )
    )

    print(
        "exact =",
        int(
            x[
                f"{name}_exact"
            ].sum()
        )
    )


# ------------------------------------------------------------
# Pairwise common-valid-wrong comparisons
# ------------------------------------------------------------

pairs = [
    ("DiffMS", "FRIGID"),
    ("DiffMS", "MS2Forge"),
    ("FRIGID", "MS2Forge"),
]


for A, B in pairs:

    subset = x[
        x[f"{A}_valid"]
        &
        x[f"{B}_valid"]
        &
        ~x[f"{A}_exact"]
        &
        ~x[f"{B}_exact"]
    ].copy()

    delta = (
        subset[
            f"{B}_tanimoto"
        ]
        -
        subset[
            f"{A}_tanimoto"
        ]
    ).to_numpy()

    boot = bootstrap(
        delta
    )

    pos, neg, ties, p = sign_test(
        delta
    )

    print()
    print("=" * 105)
    print(
        f"{A} vs {B} — COMMON VALID WRONG"
    )
    print("=" * 105)

    print(
        "N =",
        len(subset)
    )

    print(
        f"{A} mean/median =",
        subset[
            f"{A}_tanimoto"
        ].mean(),
        "/",
        subset[
            f"{A}_tanimoto"
        ].median()
    )

    print(
        f"{B} mean/median =",
        subset[
            f"{B}_tanimoto"
        ].mean(),
        "/",
        subset[
            f"{B}_tanimoto"
        ].median()
    )

    print(
        f"Mean delta {B}-{A} =",
        boot["mean"]
    )

    print(
        "95% CI mean =",
        (
            boot["mean_low"],
            boot["mean_high"],
        )
    )

    print(
        "Median delta =",
        boot["median"]
    )

    print(
        f"{B} better =",
        pos,
        "/",
        len(subset)
    )

    print(
        f"{A} better =",
        neg,
        "/",
        len(subset)
    )

    print(
        "Tie =",
        ties
    )

    print(
        "Sign-test p =",
        p
    )


# ------------------------------------------------------------
# Triple-common-valid-wrong subset
# ------------------------------------------------------------

triple = x[
    x["DiffMS_valid"]
    &
    x["FRIGID_valid"]
    &
    x["MS2Forge_valid"]
    &
    ~x["DiffMS_exact"]
    &
    ~x["FRIGID_exact"]
    &
    ~x["MS2Forge_exact"]
].copy()


print()
print("=" * 105)
print("TRIPLE COMMON-VALID-WRONG")
print("=" * 105)

print(
    "N =",
    len(triple)
)

for name in FILES:

    print(
        name,
        "mean/median =",
        triple[
            f"{name}_tanimoto"
        ].mean(),
        "/",
        triple[
            f"{name}_tanimoto"
        ].median()
    )


x.to_csv(
    OUT,
    index=False
)

print()
print(
    "Written:",
    OUT
)
