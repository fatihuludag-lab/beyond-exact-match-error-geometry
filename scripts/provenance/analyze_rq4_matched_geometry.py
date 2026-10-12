from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


REAL = Path(
    r"results\RQ4_REAL_GEOMETRY_EG100.csv"
)

SYN = Path(
    r"results\RQ4_SYNTHETIC_GEOMETRY_EG100.csv"
)

OUT = Path(
    r"results\RQ4_MATCHED_REAL_vs_SYNTHETIC.csv"
)

SEED = 20261007
N_BOOT = 20000


PRIMARY_METRICS = [
    "tanimoto",
    "mcs_order_atom_fraction",
    "mcs_order_bond_fraction",
    "mcs_topology_atom_fraction",
    "mcs_topology_bond_fraction",
]


def sign_test(values):

    x = np.asarray(
        values,
        dtype=float
    )

    pos = int((x > 0).sum())
    neg = int((x < 0).sum())
    tie = int((x == 0).sum())

    n = pos + neg

    if n == 0:
        return pos, neg, tie, 1.0

    lo = min(pos, neg)

    tail = sum(
        comb(n, i)
        for i in range(lo + 1)
    ) / (2 ** n)

    p = min(
        1.0,
        2.0 * tail
    )

    return (
        pos,
        neg,
        tie,
        p,
    )


def bootstrap(values):

    x = np.asarray(
        values,
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

    samples = x[idx]

    means = samples.mean(
        axis=1
    )

    medians = np.median(
        samples,
        axis=1
    )

    return {
        "mean":
            float(x.mean()),

        "mean_low":
            float(
                np.quantile(
                    means,
                    0.025
                )
            ),

        "mean_high":
            float(
                np.quantile(
                    means,
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
                    medians,
                    0.025
                )
            ),

        "median_high":
            float(
                np.quantile(
                    medians,
                    0.975
                )
            ),
    }


real = pd.read_csv(REAL)
syn = pd.read_csv(SYN)


if (
    real[
        [
            "mcs_order_timeout",
            "mcs_topology_timeout",
        ]
    ].any().any()
):
    raise RuntimeError(
        "Real MCS timeout detected."
    )


if (
    syn[
        [
            "mcs_order_timeout",
            "mcs_topology_timeout",
        ]
    ].any().any()
):
    raise RuntimeError(
        "Synthetic MCS timeout detected."
    )


# ------------------------------------------------------------
# Synthetic median geometry for each truth molecule.
# ------------------------------------------------------------

syn_parent = (
    syn.groupby("parent_spec")[
        PRIMARY_METRICS
    ]
    .median()
    .add_prefix("synthetic_median_")
    .reset_index()
)


# ------------------------------------------------------------
# Primary inferential unit = parent molecule.
#
# If both DiffMS and FRIGID provide a real error for the
# same parent, aggregate them BEFORE inference.
# ------------------------------------------------------------

real_parent = (
    real.groupby("parent_spec")[
        PRIMARY_METRICS
    ]
    .median()
    .add_prefix("real_median_")
    .reset_index()
)


matched = real_parent.merge(
    syn_parent,
    on="parent_spec",
    how="inner",
    validate="one_to_one",
)


if len(matched) != 88:
    raise RuntimeError(
        f"Expected 88 parents, got {len(matched)}"
    )


print("=" * 105)
print("RQ4 — MOLECULE-MATCHED REAL vs SYNTHETIC ERROR GEOMETRY")
print("=" * 105)

print(
    "PARENTS =",
    len(matched)
)


for metric in PRIMARY_METRICS:

    real_col = (
        "real_median_"
        + metric
    )

    syn_col = (
        "synthetic_median_"
        + metric
    )

    delta_col = (
        "delta_real_minus_synthetic_"
        + metric
    )


    matched[
        delta_col
    ] = (
        matched[real_col]
        -
        matched[syn_col]
    )


    delta = matched[
        delta_col
    ].to_numpy(
        dtype=float
    )


    boot = bootstrap(
        delta
    )

    pos, neg, tie, p = sign_test(
        delta
    )


    print()
    print("=" * 105)
    print(metric.upper())
    print("=" * 105)

    print(
        "Real median across parents =",
        matched[
            real_col
        ].median()
    )

    print(
        "Synthetic median across parents =",
        matched[
            syn_col
        ].median()
    )

    print()
    print(
        "Mean delta real-synthetic =",
        boot["mean"]
    )

    print(
        "95% bootstrap CI mean =",
        (
            boot["mean_low"],
            boot["mean_high"],
        )
    )

    print(
        "Median delta real-synthetic =",
        boot["median"]
    )

    print(
        "95% bootstrap CI median =",
        (
            boot["median_low"],
            boot["median_high"],
        )
    )

    print()
    print(
        "Real closer =",
        pos,
        "/",
        len(matched)
    )

    print(
        "Synthetic closer =",
        neg,
        "/",
        len(matched)
    )

    print(
        "Tie =",
        tie
    )

    print(
        "Exact sign-test p =",
        p
    )


# ------------------------------------------------------------
# Model-specific sensitivity analyses.
# Each model contributes max one observation per parent.
# ------------------------------------------------------------

for model in sorted(
    real[
        "model_or_family"
    ].unique()
):

    model_df = real[
        real[
            "model_or_family"
        ] == model
    ].copy()

    model_df = model_df.merge(
        syn_parent,
        on="parent_spec",
        how="inner",
        validate="many_to_one",
    )

    print()
    print("#" * 105)
    print(
        "MODEL-SPECIFIC SENSITIVITY:",
        model
    )
    print("#" * 105)

    print(
        "N =",
        len(model_df)
    )

    for metric in PRIMARY_METRICS:

        delta = (
            model_df[metric]
            -
            model_df[
                "synthetic_median_"
                + metric
            ]
        ).to_numpy(
            dtype=float
        )

        boot = bootstrap(
            delta
        )

        pos, neg, tie, p = sign_test(
            delta
        )

        print()
        print(metric)

        print(
            "mean delta =",
            boot["mean"]
        )

        print(
            "95% CI =",
            (
                boot["mean_low"],
                boot["mean_high"],
            )
        )

        print(
            "real closer / synthetic closer / tie =",
            pos,
            "/",
            neg,
            "/",
            tie
        )

        print(
            "sign p =",
            p
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
