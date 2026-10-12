from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


MAP = Path(
    r"results\mces\EG100_COMMON_WRONG_MCES_MAP.csv"
)

MCES = Path(
    r"results\mces\EG100_COMMON_WRONG_MCES_OUT.csv"
)

OUT = Path(
    r"results\paired_DIFFMS_vs_FRIGID_EG100_MCES.csv"
)

N_BOOT = 20000
SEED = 20261007


def exact_sign_test(delta):
    """
    MCES: lower is better.
    delta = FRIGID - DiffMS
    delta < 0 => FRIGID better
    """

    delta = np.asarray(delta, dtype=float)

    wins_f = int((delta < 0).sum())
    wins_d = int((delta > 0).sum())
    ties = int((delta == 0).sum())

    n = wins_f + wins_d

    if n == 0:
        return wins_f, wins_d, ties, 1.0

    lo = min(wins_f, wins_d)

    tail = sum(
        comb(n, i)
        for i in range(lo + 1)
    ) / (2 ** n)

    p = min(
        1.0,
        2.0 * tail
    )

    return (
        wins_f,
        wins_d,
        ties,
        p,
    )


def paired_bootstrap(delta):

    delta = np.asarray(
        delta,
        dtype=float
    )

    rng = np.random.default_rng(
        SEED
    )

    idx = rng.integers(
        0,
        len(delta),
        size=(
            N_BOOT,
            len(delta)
        ),
    )

    samples = delta[idx]

    mean_boot = samples.mean(
        axis=1
    )

    median_boot = np.median(
        samples,
        axis=1
    )

    return {
        "mean":
            float(delta.mean()),

        "mean_ci_low":
            float(
                np.quantile(
                    mean_boot,
                    0.025
                )
            ),

        "mean_ci_high":
            float(
                np.quantile(
                    mean_boot,
                    0.975
                )
            ),

        "median":
            float(
                np.median(delta)
            ),

        "median_ci_low":
            float(
                np.quantile(
                    median_boot,
                    0.025
                )
            ),

        "median_ci_high":
            float(
                np.quantile(
                    median_boot,
                    0.975
                )
            ),
    }


def report(label, df):

    print()
    print("=" * 105)
    print(label)
    print("=" * 105)

    print(
        "N =",
        len(df)
    )

    if len(df) == 0:
        return

    print()
    print(
        "DiffMS MCES mean/median =",
        df["diffms_mces"].mean(),
        "/",
        df["diffms_mces"].median()
    )

    print(
        "FRIGID MCES mean/median =",
        df["frigid_mces"].mean(),
        "/",
        df["frigid_mces"].median()
    )

    delta = (
        df[
            "delta_F_minus_D"
        ]
        .to_numpy(
            dtype=float
        )
    )

    boot = paired_bootstrap(
        delta
    )

    (
        wins_f,
        wins_d,
        ties,
        p_sign,
    ) = exact_sign_test(
        delta
    )

    print()
    print(
        "Mean delta F-D =",
        boot["mean"]
    )

    print(
        "95% bootstrap CI mean =",
        (
            boot["mean_ci_low"],
            boot["mean_ci_high"],
        )
    )

    print(
        "Median delta F-D =",
        boot["median"]
    )

    print(
        "95% bootstrap CI median =",
        (
            boot["median_ci_low"],
            boot["median_ci_high"],
        )
    )

    print()
    print(
        "FRIGID better =",
        wins_f,
        "/",
        len(df)
    )

    print(
        "DiffMS better =",
        wins_d,
        "/",
        len(df)
    )

    print(
        "Tie =",
        ties,
        "/",
        len(df)
    )

    print(
        "Exact sign-test p =",
        p_sign
    )

    print()
    print(
        "P(FRIGID better | non-tie) =",
        (
            wins_f
            /
            (wins_f + wins_d)
            if (wins_f + wins_d)
            else None
        )
    )


# ============================================================
# LOAD
# ============================================================

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


if len(mapping) != 92:
    raise RuntimeError(
        f"Expected 92 map rows, got {len(mapping)}"
    )

if len(dist) != 92:
    raise RuntimeError(
        f"Expected 92 MCES rows, got {len(dist)}"
    )

if dist["pair_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate MCES pair IDs."
    )

if (dist["mces"] < 0).any():
    bad = dist[
        dist["mces"] < 0
    ]

    print(bad)

    raise RuntimeError(
        "Failed MCES calculations detected."
    )


long = mapping.merge(
    dist,
    on="pair_id",
    how="inner",
    validate="one_to_one",
)


# ============================================================
# WIDE PAIRED TABLE
# ============================================================

distance_wide = (
    long.pivot(
        index=[
            "panel_index",
            "spec",
        ],
        columns="model",
        values="mces",
    )
    .reset_index()
)

distance_wide.columns.name = None

distance_wide = distance_wide.rename(
    columns={
        "DiffMS-K1":
            "diffms_mces",

        "FRIGID-base@FM1":
            "frigid_mces",
    }
)


# Formula flags
formula_wide = (
    long.pivot(
        index=[
            "panel_index",
            "spec",
        ],
        columns="model",
        values="same_formula",
    )
    .reset_index()
)

formula_wide.columns.name = None

formula_wide = formula_wide.rename(
    columns={
        "DiffMS-K1":
            "diffms_same_formula",

        "FRIGID-base@FM1":
            "frigid_same_formula",
    }
)


# Tanimoto for cross-metric comparison
tani_wide = (
    long.pivot(
        index=[
            "panel_index",
            "spec",
        ],
        columns="model",
        values="tanimoto",
    )
    .reset_index()
)

tani_wide.columns.name = None

tani_wide = tani_wide.rename(
    columns={
        "DiffMS-K1":
            "diffms_tanimoto",

        "FRIGID-base@FM1":
            "frigid_tanimoto",
    }
)


wide = (
    distance_wide
    .merge(
        formula_wide,
        on=[
            "panel_index",
            "spec",
        ],
        validate="one_to_one",
    )
    .merge(
        tani_wide,
        on=[
            "panel_index",
            "spec",
        ],
        validate="one_to_one",
    )
)


if len(wide) != 46:
    raise RuntimeError(
        f"Expected 46 paired molecules, got {len(wide)}"
    )


# MCES: lower = better.
wide[
    "delta_F_minus_D"
] = (
    wide[
        "frigid_mces"
    ]
    -
    wide[
        "diffms_mces"
    ]
)


wide[
    "frigid_better_mces"
] = (
    wide[
        "frigid_mces"
    ]
    <
    wide[
        "diffms_mces"
    ]
)

wide[
    "diffms_better_mces"
] = (
    wide[
        "diffms_mces"
    ]
    <
    wide[
        "frigid_mces"
    ]
)

wide["mces_tie"] = (
    wide[
        "diffms_mces"
    ]
    ==
    wide[
        "frigid_mces"
    ]
)


# ============================================================
# MAIN ANALYSIS
# ============================================================

print("=" * 105)
print("PAIRED MCES — DIFFMS-K1 vs FRIGID-base@FM1")
print("=" * 105)

print(
    "MCES pairs =",
    len(long)
)

print(
    "Molecules =",
    len(wide)
)

print(
    "Failed MCES =",
    int(
        (long["mces"] < 0).sum()
    )
)

print(
    "MCES mode counts =",
    long[
        "mces_mode"
    ].value_counts().to_dict()
)

print()
print(
    "Mean MCES runtime sec =",
    long[
        "mces_time_sec"
    ].mean()
)

print(
    "Median MCES runtime sec =",
    long[
        "mces_time_sec"
    ].median()
)

print(
    "Max MCES runtime sec =",
    long[
        "mces_time_sec"
    ].max()
)


report(
    "ALL COMMON-VALID-WRONG",
    wide
)


# ============================================================
# BOTH-SAME-FORMULA SENSITIVITY ANALYSIS
# ============================================================

both_formula = wide[
    wide[
        "diffms_same_formula"
    ].astype(bool)
    &
    wide[
        "frigid_same_formula"
    ].astype(bool)
].copy()


report(
    "BOTH-SAME-FORMULA COMMON-WRONG",
    both_formula
)


# ============================================================
# RELATION TO TANIMOTO
# ============================================================

wide[
    "delta_tanimoto_F_minus_D"
] = (
    wide[
        "frigid_tanimoto"
    ]
    -
    wide[
        "diffms_tanimoto"
    ]
)


corr = np.corrcoef(
    wide[
        "delta_tanimoto_F_minus_D"
    ],
    -wide[
        "delta_F_minus_D"
    ],
)[0, 1]


print()
print("=" * 105)
print("MCES vs TANIMOTO IMPROVEMENT")
print("=" * 105)

print(
    "Correlation between "
    "FRIGID Tanimoto gain and "
    "FRIGID MCES improvement =",
    corr
)

print(
    "(MCES improvement is defined as DiffMS MCES - FRIGID MCES)"
)


# ============================================================
# SAVE
# ============================================================

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

wide.to_csv(
    OUT,
    index=False
)

print()
print("Written:", OUT)
