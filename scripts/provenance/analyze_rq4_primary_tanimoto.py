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
    r"results\RQ4_PRIMARY_MATCHED_TANIMOTO.csv"
)

SEED = 20261008
N_BOOT = 50000


def sign_test(x):

    x = np.asarray(x, dtype=float)

    pos = int((x > 0).sum())
    neg = int((x < 0).sum())
    ties = int((x == 0).sum())

    n = pos + neg

    if n == 0:
        return pos, neg, ties, 1.0

    lo = min(pos, neg)

    tail = (
        sum(comb(n, i) for i in range(lo + 1))
        / 2**n
    )

    return (
        pos,
        neg,
        ties,
        min(1.0, 2 * tail),
    )


def bootstrap(x):

    x = np.asarray(x, dtype=float)

    rng = np.random.default_rng(SEED)

    ids = rng.integers(
        0,
        len(x),
        size=(N_BOOT, len(x)),
    )

    z = x[ids]

    return {
        "mean": float(x.mean()),
        "mean_low":
            float(np.quantile(z.mean(1), .025)),
        "mean_high":
            float(np.quantile(z.mean(1), .975)),

        "median": float(np.median(x)),
        "median_low":
            float(
                np.quantile(
                    np.median(z, axis=1),
                    .025
                )
            ),
        "median_high":
            float(
                np.quantile(
                    np.median(z, axis=1),
                    .975
                )
            ),
    }


real = pd.read_csv(REAL)
syn = pd.read_csv(SYN)

assert len(real) == 98
assert len(syn) == 860

assert real["same_formula"].all()
assert syn["same_formula"].all()


# ============================================================
# Synthetic geometry:
# one median summary per truth molecule.
# ============================================================

syn_parent = (
    syn.groupby("parent_spec")
    .agg(
        synthetic_median_tanimoto=(
            "tanimoto",
            "median"
        ),
        synthetic_mean_tanimoto=(
            "tanimoto",
            "mean"
        ),
        n_synthetic=(
            "observation_id",
            "size"
        ),
    )
    .reset_index()
)


# ============================================================
# Real geometry:
# 10 parents have two real generator errors.
# Aggregate before primary inference so the unit is molecule.
# ============================================================

real_parent = (
    real.groupby("parent_spec")
    .agg(
        real_median_tanimoto=(
            "tanimoto",
            "median"
        ),
        real_mean_tanimoto=(
            "tanimoto",
            "mean"
        ),
        n_real_errors=(
            "observation_id",
            "size"
        ),
    )
    .reset_index()
)


matched = real_parent.merge(
    syn_parent,
    on="parent_spec",
    validate="one_to_one",
)


if len(matched) != 88:
    raise RuntimeError(
        f"Expected 88 parents, got {len(matched)}"
    )


matched["delta_real_minus_synthetic"] = (
    matched["real_median_tanimoto"]
    -
    matched["synthetic_median_tanimoto"]
)


delta = matched[
    "delta_real_minus_synthetic"
].to_numpy()


boot = bootstrap(delta)

pos, neg, ties, p = sign_test(delta)


print("=" * 105)
print("RQ4 PRIMARY — MOLECULE-MATCHED TANIMOTO")
print("=" * 105)

print("PARENTS =", len(matched))

print()
print(
    "Real median across parents =",
    matched[
        "real_median_tanimoto"
    ].median()
)

print(
    "Synthetic median across parents =",
    matched[
        "synthetic_median_tanimoto"
    ].median()
)

print()
print(
    "Mean paired delta real-synthetic =",
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
    "Median paired delta =",
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
    "REAL closer =",
    pos,
    "/",
    len(matched)
)

print(
    "SYNTHETIC closer =",
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


# Paired standardized effect.
sd = delta.std(ddof=1)

print(
    "Paired standardized effect dz =",
    delta.mean() / sd
    if sd > 0 else None
)


# ============================================================
# Model-specific sensitivity.
# ============================================================

for model in sorted(
    real["model_or_family"].unique()
):

    g = real[
        real["model_or_family"] == model
    ].merge(
        syn_parent,
        on="parent_spec",
        validate="many_to_one",
    )

    d = (
        g["tanimoto"]
        -
        g["synthetic_median_tanimoto"]
    ).to_numpy()

    b = bootstrap(d)

    po, ne, ti, pv = sign_test(d)

    print()
    print("-" * 105)
    print(
        "MODEL-SPECIFIC:",
        model
    )
    print("-" * 105)

    print("N =", len(g))

    print(
        "Real Tanimoto median =",
        g["tanimoto"].median()
    )

    print(
        "Synthetic matched median =",
        g[
            "synthetic_median_tanimoto"
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
        "real closer / synthetic closer / tie =",
        po, "/", ne, "/", ti
    )

    print(
        "sign p =",
        pv
    )


matched.to_csv(
    OUT,
    index=False
)

print()
print("Written:", OUT)
