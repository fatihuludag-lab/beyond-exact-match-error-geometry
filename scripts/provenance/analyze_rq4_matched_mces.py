from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


MAP = Path(
    r"results\mces\RQ4_MATCHED_MCES_MAP.csv"
)

MCES = Path(
    r"results\mces\RQ4_MATCHED_MCES_OUT.csv"
)

OUT = Path(
    r"results\RQ4_MATCHED_MCES_ANALYSIS.csv"
)

N_BOOT = 50000
SEED = 20261008


def exact_sign_test(x):

    x = np.asarray(x, dtype=float)

    pos = int((x > 0).sum())
    neg = int((x < 0).sum())
    ties = int((x == 0).sum())

    n = pos + neg

    if n == 0:
        return pos, neg, ties, 1.0

    lo = min(pos, neg)

    tail = (
        sum(
            comb(n, i)
            for i in range(lo + 1)
        )
        / (2 ** n)
    )

    return (
        pos,
        neg,
        ties,
        min(1.0, 2.0 * tail),
    )


def bootstrap(x):

    x = np.asarray(x, dtype=float)

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

    mean_boot = z.mean(axis=1)
    median_boot = np.median(
        z,
        axis=1
    )

    return {
        "mean":
            float(x.mean()),

        "mean_low":
            float(
                np.quantile(
                    mean_boot,
                    0.025
                )
            ),

        "mean_high":
            float(
                np.quantile(
                    mean_boot,
                    0.975
                )
            ),

        "median":
            float(
                np.median(x)
            ),

        "median_low":
            float(
                np.quantile(
                    median_boot,
                    0.025
                )
            ),

        "median_high":
            float(
                np.quantile(
                    median_boot,
                    0.975
                )
            ),
    }


mapping = pd.read_csv(
    MAP
)

dist = pd.read_csv(
    MCES,
    header=None,
    names=[
        "pair_id",
        "mces",
        "mces_time_sec",
        "mces_mode",
    ],
)


# ============================================================
# INTEGRITY
# ============================================================

if len(mapping) != 186:
    raise RuntimeError(
        f"Expected 186 map rows, got {len(mapping)}"
    )

if len(dist) != 186:
    raise RuntimeError(
        f"Expected 186 MCES rows, got {len(dist)}"
    )

if dist["pair_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate MCES pair IDs."
    )

if (dist["mces"] < 0).any():
    raise RuntimeError(
        "Failed MCES computation detected."
    )


long = mapping.merge(
    dist,
    on="pair_id",
    validate="one_to_one",
)


print("=" * 105)
print("RQ4 — REAL vs SYNTHETIC MCES")
print("=" * 105)

print(
    "TOTAL PAIRS =",
    len(long)
)

print()
print(
    long["source"]
    .value_counts()
    .to_string()
)

print()
print(
    "MCES mode =",
    long["mces_mode"]
    .value_counts()
    .to_dict()
)

print(
    "Mean runtime sec =",
    long["mces_time_sec"].mean()
)

print(
    "Median runtime sec =",
    long["mces_time_sec"].median()
)

print(
    "Max runtime sec =",
    long["mces_time_sec"].max()
)


# ============================================================
# RAW DESCRIPTIVE DISTRIBUTIONS
# ============================================================

for source in [
    "real",
    "synthetic",
]:

    g = long[
        long["source"] == source
    ]

    print()
    print(source.upper())

    print(
        "N =",
        len(g)
    )

    print(
        "MCES mean/median =",
        g["mces"].mean(),
        "/",
        g["mces"].median()
    )

    print(
        "Tanimoto mean/median =",
        g["tanimoto"].mean(),
        "/",
        g["tanimoto"].median()
    )


# ============================================================
# PARENT-LEVEL PRIMARY MCES CONFIRMATION
#
# Real:
# 10 parents have two generator errors.
# Aggregate them before inference.
#
# Synthetic:
# exactly one representative corruption per parent.
# ============================================================

real = (
    long[
        long["source"] == "real"
    ]
    .groupby("parent_spec")
    .agg(
        real_median_mces=(
            "mces",
            "median"
        ),

        real_median_tanimoto=(
            "tanimoto",
            "median"
        ),

        n_real_errors=(
            "pair_id",
            "size"
        ),
    )
    .reset_index()
)


synthetic = (
    long[
        long["source"] == "synthetic"
    ]
    [
        [
            "parent_spec",
            "mces",
            "tanimoto",
            "subgroup",
        ]
    ]
    .rename(
        columns={
            "mces":
                "synthetic_mces",

            "tanimoto":
                "synthetic_tanimoto",

            "subgroup":
                "synthetic_family",
        }
    )
)


if synthetic[
    "parent_spec"
].duplicated().any():

    raise RuntimeError(
        "Synthetic representative is not unique per parent."
    )


matched = real.merge(
    synthetic,
    on="parent_spec",
    validate="one_to_one",
)


if len(matched) != 88:
    raise RuntimeError(
        f"Expected 88 parents, got {len(matched)}"
    )


# MCES: HIGHER = FARTHER.
#
# Positive delta means real errors are farther than
# the matched synthetic representative.

matched[
    "delta_mces_real_minus_synthetic"
] = (
    matched[
        "real_median_mces"
    ]
    -
    matched[
        "synthetic_mces"
    ]
)


delta = matched[
    "delta_mces_real_minus_synthetic"
].to_numpy(
    dtype=float
)


boot = bootstrap(
    delta
)

pos, neg, ties, p = (
    exact_sign_test(delta)
)


print()
print("=" * 105)
print("PARENT-MATCHED MCES RESULT")
print("=" * 105)

print(
    "PARENTS =",
    len(matched)
)

print()
print(
    "Real MCES median across parents =",
    matched[
        "real_median_mces"
    ].median()
)

print(
    "Synthetic MCES median across parents =",
    matched[
        "synthetic_mces"
    ].median()
)

print()
print(
    "Mean delta real-synthetic =",
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
    "95% CI median =",
    (
        boot["median_low"],
        boot["median_high"],
    )
)

print()
print(
    "REAL farther =",
    pos,
    "/",
    len(matched)
)

print(
    "SYNTHETIC farther =",
    neg,
    "/",
    len(matched)
)

print(
    "Tie =",
    ties
)

print(
    "Exact sign-test p =",
    p
)


sd = delta.std(
    ddof=1
)

print(
    "Paired standardized effect dz =",
    (
        delta.mean() / sd
        if sd > 0
        else None
    )
)


# ============================================================
# RELATION BETWEEN TANIMOTO GAP AND MCES GAP
# ============================================================

matched[
    "tanimoto_gap_synthetic_minus_real"
] = (
    matched[
        "synthetic_tanimoto"
    ]
    -
    matched[
        "real_median_tanimoto"
    ]
)


corr = np.corrcoef(
    matched[
        "tanimoto_gap_synthetic_minus_real"
    ],
    matched[
        "delta_mces_real_minus_synthetic"
    ],
)[0, 1]


print()
print("=" * 105)
print("TANIMOTO GAP vs MCES GAP")
print("=" * 105)

print(
    "Correlation =",
    corr
)


# ============================================================
# MODEL-SPECIFIC SENSITIVITY
# ============================================================

real_long = long[
    long["source"] == "real"
].copy()


for model in sorted(
    real_long[
        "subgroup"
    ].unique()
):

    g = real_long[
        real_long[
            "subgroup"
        ] == model
    ].merge(
        synthetic[
            [
                "parent_spec",
                "synthetic_mces",
            ]
        ],
        on="parent_spec",
        validate="many_to_one",
    )

    d = (
        g["mces"]
        -
        g[
            "synthetic_mces"
        ]
    ).to_numpy(
        dtype=float
    )

    b = bootstrap(d)

    po, ne, ti, pv = (
        exact_sign_test(d)
    )

    print()
    print("-" * 105)
    print(
        "MODEL-SPECIFIC:",
        model
    )
    print("-" * 105)

    print(
        "N =",
        len(g)
    )

    print(
        "Real MCES median =",
        g["mces"].median()
    )

    print(
        "Synthetic MCES median =",
        g[
            "synthetic_mces"
        ].median()
    )

    print(
        "Mean delta =",
        b["mean"]
    )

    print(
        "95% CI =",
        (
            b["mean_low"],
            b["mean_high"],
        )
    )

    print(
        "real farther / synthetic farther / tie =",
        po, "/", ne, "/", ti
    )

    print(
        "sign p =",
        pv
    )


OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

matched.to_csv(
    OUT,
    index=False
)

print()
print("Written:", OUT)
