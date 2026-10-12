from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


# Real MCES results already computed in the 186-pair run.
REAL_MAP = Path(
    r"results\mces\RQ4_MATCHED_MCES_MAP.csv"
)

REAL_OUT = Path(
    r"results\mces\RQ4_MATCHED_MCES_OUT.csv"
)

# All 860 primary synthetic corruptions.
SYN_MAP = Path(
    r"results\mces\RQ4_SYNTHETIC_ALL860_MAP.csv"
)

SYN_OUT = Path(
    r"results\mces\RQ4_SYNTHETIC_ALL860_OUT.csv"
)

OUT = Path(
    r"results\RQ4_FINAL_ALL_SYNTHETIC_MCES.csv"
)

FAMILY_OUT = Path(
    r"results\RQ4_FINAL_FAMILY_SENSITIVITY_MCES.csv"
)

N_BOOT = 50000
SEED = 20261008


def sign_test(x):

    x = np.asarray(x, dtype=float)

    pos = int((x > 0).sum())
    neg = int((x < 0).sum())
    tie = int((x == 0).sum())

    n = pos + neg

    if n == 0:
        return pos, neg, tie, 1.0

    lo = min(pos, neg)

    tail = (
        sum(
            comb(n, i)
            for i in range(lo + 1)
        )
        / (2 ** n)
    )

    p = min(
        1.0,
        2.0 * tail
    )

    return pos, neg, tie, p


def bootstrap(x):

    x = np.asarray(
        x,
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

    means = z.mean(axis=1)

    medians = np.median(
        z,
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


def load_mces(map_path, out_path):

    mapping = pd.read_csv(
        map_path
    )

    result = pd.read_csv(
        out_path,
        header=None,
        names=[
            "pair_id",
            "mces",
            "time_sec",
            "mode",
        ],
    )

    if result["pair_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate pair IDs: {out_path}"
        )

    if (result["mces"] < 0).any():
        raise RuntimeError(
            f"Failed MCES: {out_path}"
        )

    return mapping.merge(
        result,
        on="pair_id",
        validate="one_to_one",
    )


# ============================================================
# LOAD REAL
# ============================================================

real_all = load_mces(
    REAL_MAP,
    REAL_OUT
)

real = real_all[
    real_all["source"] == "real"
].copy()


if len(real) != 98:
    raise RuntimeError(
        f"Expected 98 real errors, got {len(real)}"
    )


# ============================================================
# LOAD ALL SYNTHETIC
# ============================================================

syn = load_mces(
    SYN_MAP,
    SYN_OUT
)


if len(syn) != 860:
    raise RuntimeError(
        f"Expected 860 synthetic errors, got {len(syn)}"
    )


if real["parent_spec"].nunique() != 88:
    raise RuntimeError(
        "Real parent count != 88"
    )

if syn["parent_spec"].nunique() != 88:
    raise RuntimeError(
        "Synthetic parent count != 88"
    )


print("=" * 110)
print("RQ4 FINAL — ALL-SYNTHETIC MCES ROBUSTNESS")
print("=" * 110)

print(
    "REAL observations =",
    len(real)
)

print(
    "SYN observations =",
    len(syn)
)

print(
    "REAL parents =",
    real["parent_spec"].nunique()
)

print(
    "SYN parents =",
    syn["parent_spec"].nunique()
)


print()
print("RAW DISTRIBUTIONS")

print(
    "Real MCES mean/median =",
    real["mces"].mean(),
    "/",
    real["mces"].median()
)

print(
    "Synthetic MCES mean/median =",
    syn["mces"].mean(),
    "/",
    syn["mces"].median()
)


# ============================================================
# PARENT-LEVEL AGGREGATION
# ============================================================

# For 10 parents both generators contribute a real error.
# Aggregate to one real geometry value per molecule.

real_parent = (
    real.groupby("parent_spec")
    .agg(
        real_median_mces=(
            "mces",
            "median"
        ),

        real_mean_mces=(
            "mces",
            "mean"
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


# Use ALL primary synthetic corruptions for each molecule.

syn_parent = (
    syn.groupby("parent_spec")
    .agg(
        synthetic_median_mces=(
            "mces",
            "median"
        ),

        synthetic_mean_mces=(
            "mces",
            "mean"
        ),

        synthetic_median_tanimoto=(
            "tanimoto",
            "median"
        ),

        n_synthetic=(
            "pair_id",
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
        f"Expected 88 matched parents, got {len(matched)}"
    )


matched[
    "delta_mces_real_minus_synthetic"
] = (
    matched[
        "real_median_mces"
    ]
    -
    matched[
        "synthetic_median_mces"
    ]
)


matched[
    "delta_tanimoto_real_minus_synthetic"
] = (
    matched[
        "real_median_tanimoto"
    ]
    -
    matched[
        "synthetic_median_tanimoto"
    ]
)


delta = matched[
    "delta_mces_real_minus_synthetic"
].to_numpy(
    dtype=float
)


boot = bootstrap(delta)

pos, neg, tie, p = sign_test(
    delta
)


print()
print("=" * 110)
print("PARENT-MATCHED PRIMARY MCES RESULT")
print("=" * 110)

print(
    "PARENTS =",
    len(matched)
)

print()
print(
    "Real median MCES across parents =",
    matched[
        "real_median_mces"
    ].median()
)

print(
    "Synthetic median MCES across parents =",
    matched[
        "synthetic_median_mces"
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
    tie
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
# TANIMOTO–MCES GAP CONSISTENCY
# ============================================================

# Synthetic closer means:
# MCES real-synthetic > 0
# Tanimoto real-synthetic < 0

corr = np.corrcoef(
    matched[
        "delta_mces_real_minus_synthetic"
    ],
    -matched[
        "delta_tanimoto_real_minus_synthetic"
    ],
)[0, 1]


print()
print("=" * 110)
print("MCES GAP vs TANIMOTO GAP")
print("=" * 110)

print(
    "Correlation =",
    corr
)


# ============================================================
# MODEL-SPECIFIC REAL ERROR SENSITIVITY
# ============================================================

for model in sorted(
    real["subgroup"].unique()
):

    g = real[
        real["subgroup"] == model
    ].merge(
        syn_parent[
            [
                "parent_spec",
                "synthetic_median_mces",
            ]
        ],
        on="parent_spec",
        validate="many_to_one",
    )


    d = (
        g["mces"]
        -
        g[
            "synthetic_median_mces"
        ]
    ).to_numpy(
        dtype=float
    )


    b = bootstrap(d)

    po, ne, ti, pv = sign_test(d)


    print()
    print("-" * 110)
    print(
        "MODEL-SPECIFIC:",
        model
    )
    print("-" * 110)

    print(
        "N =",
        len(g)
    )

    print(
        "Real MCES median =",
        g["mces"].median()
    )

    print(
        "Synthetic parent-median MCES =",
        g[
            "synthetic_median_mces"
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
        po,
        "/",
        ne,
        "/",
        ti
    )

    print(
        "sign p =",
        pv
    )


# ============================================================
# CORRUPTION-FAMILY SENSITIVITY
# ============================================================

family_rows = []


for family in sorted(
    syn[
        "corruption_family"
    ].unique()
):

    sf = syn[
        syn[
            "corruption_family"
        ] == family
    ].copy()


    sf_parent = (
        sf.groupby("parent_spec")
        .agg(
            family_median_mces=(
                "mces",
                "median"
            ),
            n_family=(
                "pair_id",
                "size"
            ),
        )
        .reset_index()
    )


    q = real_parent.merge(
        sf_parent,
        on="parent_spec",
        how="inner",
        validate="one_to_one",
    )


    d = (
        q[
            "real_median_mces"
        ]
        -
        q[
            "family_median_mces"
        ]
    ).to_numpy(
        dtype=float
    )


    b = bootstrap(d)

    po, ne, ti, pv = sign_test(d)


    family_rows.append({
        "family":
            family,

        "n_parents":
            len(q),

        "real_median_mces":
            q[
                "real_median_mces"
            ].median(),

        "synthetic_family_median_mces":
            q[
                "family_median_mces"
            ].median(),

        "mean_delta":
            b["mean"],

        "mean_ci_low":
            b["mean_low"],

        "mean_ci_high":
            b["mean_high"],

        "real_farther":
            po,

        "synthetic_farther":
            ne,

        "ties":
            ti,

        "sign_p":
            pv,
    })


family_df = pd.DataFrame(
    family_rows
)


print()
print("=" * 110)
print("CORRUPTION-FAMILY SENSITIVITY")
print("=" * 110)

print(
    family_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

matched.to_csv(
    OUT,
    index=False
)

family_df.to_csv(
    FAMILY_OUT,
    index=False
)


print()
print("MAIN   =", OUT)
print("FAMILY =", FAMILY_OUT)
