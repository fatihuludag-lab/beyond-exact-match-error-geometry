from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


MAP = Path(
    r"results\mces\THREE_MODEL_NEW_PAIRS_MAP.csv"
)

MCES = Path(
    r"results\mces\THREE_MODEL_NEW_PAIRS_OUT.csv"
)

COMP = Path(
    r"results\mces\THREE_MODEL_NEW_COMPARISONS.csv"
)

OUT = Path(
    r"results\THREE_MODEL_MCES_EG100_ANALYSIS.csv"
)

N_BOOT = 50000
SEED = 20261008


def sign_test(delta):
    """
    delta = B - A
    MCES: lower is better.
    delta < 0 => B better.
    """

    x = np.asarray(delta, dtype=float)

    b_better = int((x < 0).sum())
    a_better = int((x > 0).sum())
    ties = int((x == 0).sum())

    n = b_better + a_better

    if n == 0:
        return b_better, a_better, ties, 1.0

    lo = min(
        b_better,
        a_better
    )

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

    return (
        b_better,
        a_better,
        ties,
        p
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

    mean_b = z.mean(axis=1)

    med_b = np.median(
        z,
        axis=1
    )

    return {
        "mean":
            float(x.mean()),

        "mean_low":
            float(
                np.quantile(
                    mean_b,
                    0.025
                )
            ),

        "mean_high":
            float(
                np.quantile(
                    mean_b,
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
                    med_b,
                    0.025
                )
            ),

        "median_high":
            float(
                np.quantile(
                    med_b,
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
        "time_sec",
        "mode",
    ]
)

comp = pd.read_csv(
    COMP
)


if len(dist) != 136:
    raise RuntimeError(
        f"Expected 136 MCES rows, got {len(dist)}"
    )

if (dist["mces"] < 0).any():
    raise RuntimeError(
        "Failed MCES calculations present."
    )

if dist["pair_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate pair IDs."
    )


long = mapping.merge(
    dist,
    on="pair_id",
    validate="one_to_one"
)


print("=" * 110)
print("THREE-MODEL MCES — EG100")
print("=" * 110)

print(
    "UNIQUE MCES PAIRS =",
    len(long)
)

print(
    "MCES mode counts =",
    long["mode"]
    .value_counts()
    .to_dict()
)

print(
    "Mean runtime =",
    long["time_sec"].mean()
)

print(
    "Median runtime =",
    long["time_sec"].median()
)

print(
    "Max runtime =",
    long["time_sec"].max()
)


results = []


# ============================================================
# PAIRWISE COMPARISONS
# ============================================================

for comparison in [
    "DiffMS_vs_MS2Forge",
    "FRIGID_vs_MS2Forge",
]:

    c = comp[
        comp["comparison"]
        == comparison
    ].copy()

    A = c["model_A"].iloc[0]
    B = c["model_B"].iloc[0]

    specs = set(
        c["spec"]
    )

    z = long[
        long["spec"].isin(specs)
        &
        long["model"].isin(
            [A, B]
        )
    ].copy()


    dist_wide = (
        z.pivot(
            index=[
                "panel_index",
                "spec",
            ],
            columns="model",
            values="mces",
        )
        .reset_index()
    )

    dist_wide.columns.name = None


    formula_wide = (
        z.pivot(
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


    if len(dist_wide) != len(c):
        raise RuntimeError(
            f"{comparison}: "
            f"expected {len(c)}, "
            f"got {len(dist_wide)}"
        )


    delta = (
        dist_wide[B]
        -
        dist_wide[A]
    ).to_numpy(
        dtype=float
    )


    boot = bootstrap(
        delta
    )

    (
        b_better,
        a_better,
        ties,
        p
    ) = sign_test(
        delta
    )


    print()
    print("=" * 110)
    print(
        comparison,
        "— COMMON VALID WRONG"
    )
    print("=" * 110)

    print(
        "N =",
        len(dist_wide)
    )

    print(
        f"{A} MCES mean/median =",
        dist_wide[A].mean(),
        "/",
        dist_wide[A].median()
    )

    print(
        f"{B} MCES mean/median =",
        dist_wide[B].mean(),
        "/",
        dist_wide[B].median()
    )

    print()
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
        "95% CI median =",
        (
            boot["median_low"],
            boot["median_high"],
        )
    )

    print()
    print(
        f"{B} better =",
        b_better,
        "/",
        len(dist_wide)
    )

    print(
        f"{A} better =",
        a_better,
        "/",
        len(dist_wide)
    )

    print(
        "Tie =",
        ties
    )

    print(
        "Exact sign-test p =",
        p
    )


    sd = delta.std(ddof=1)

    print(
        "Paired dz =",
        (
            delta.mean() / sd
            if sd > 0
            else None
        )
    )


    # --------------------------------------------------------
    # BOTH-SAME-FORMULA sensitivity
    # --------------------------------------------------------

    both_formula_specs = formula_wide[
        formula_wide[A].astype(bool)
        &
        formula_wide[B].astype(bool)
    ][
        ["panel_index", "spec"]
    ]


    sf = dist_wide.merge(
        both_formula_specs,
        on=[
            "panel_index",
            "spec",
        ],
        validate="one_to_one",
    )


    print()
    print(
        "BOTH-SAME-FORMULA N =",
        len(sf)
    )

    if len(sf):

        d_sf = (
            sf[B]
            -
            sf[A]
        ).to_numpy(
            dtype=float
        )

        bsf = bootstrap(
            d_sf
        )

        (
            bb,
            aa,
            tt,
            pp
        ) = sign_test(
            d_sf
        )

        print(
            f"{A} median =",
            sf[A].median()
        )

        print(
            f"{B} median =",
            sf[B].median()
        )

        print(
            f"Mean delta {B}-{A} =",
            bsf["mean"]
        )

        print(
            "95% CI mean =",
            (
                bsf["mean_low"],
                bsf["mean_high"],
            )
        )

        print(
            f"{B} better / "
            f"{A} better / tie =",
            bb,
            "/",
            aa,
            "/",
            tt
        )

        print(
            "Sign-test p =",
            pp
        )


    results.append({
        "comparison":
            comparison,

        "n":
            len(dist_wide),

        "model_A":
            A,

        "model_B":
            B,

        "A_mean_mces":
            dist_wide[A].mean(),

        "B_mean_mces":
            dist_wide[B].mean(),

        "A_median_mces":
            dist_wide[A].median(),

        "B_median_mces":
            dist_wide[B].median(),

        "mean_delta_B_minus_A":
            boot["mean"],

        "mean_ci_low":
            boot["mean_low"],

        "mean_ci_high":
            boot["mean_high"],

        "B_better":
            b_better,

        "A_better":
            a_better,

        "ties":
            ties,

        "sign_p":
            p,

        "both_formula_n":
            len(sf),
    })


# ============================================================
# TRIPLE COMMON-VALID-WRONG
# ============================================================

specs_D_M = set(
    comp.loc[
        comp["comparison"]
        == "DiffMS_vs_MS2Forge",
        "spec"
    ]
)

specs_F_M = set(
    comp.loc[
        comp["comparison"]
        == "FRIGID_vs_MS2Forge",
        "spec"
    ]
)

triple_specs = (
    specs_D_M
    &
    specs_F_M
)


triple_long = long[
    long["spec"].isin(
        triple_specs
    )
].copy()


triple = (
    triple_long.pivot(
        index=[
            "panel_index",
            "spec",
        ],
        columns="model",
        values="mces",
    )
    .reset_index()
)

triple.columns.name = None


required = {
    "DiffMS",
    "FRIGID",
    "MS2Forge",
}

if not required.issubset(
    set(triple.columns)
):
    raise RuntimeError(
        f"Missing models in triple table: "
        f"{set(triple.columns)}"
    )


print()
print("=" * 110)
print("TRIPLE COMMON-VALID-WRONG MCES")
print("=" * 110)

print(
    "N =",
    len(triple)
)

for model in [
    "DiffMS",
    "FRIGID",
    "MS2Forge",
]:

    print(
        model,
        "mean/median =",
        triple[model].mean(),
        "/",
        triple[model].median()
    )


# Winner count: lowest MCES.
winner_counts = {
    "DiffMS": 0,
    "FRIGID": 0,
    "MS2Forge": 0,
    "tie": 0,
}


for r in triple.itertuples(index=False):

    vals = {
        "DiffMS":
            r.DiffMS,

        "FRIGID":
            r.FRIGID,

        "MS2Forge":
            r.MS2Forge,
    }

    minimum = min(
        vals.values()
    )

    winners = [
        k
        for k, v in vals.items()
        if v == minimum
    ]

    if len(winners) == 1:
        winner_counts[
            winners[0]
        ] += 1
    else:
        winner_counts[
            "tie"
        ] += 1


print()
print(
    "Triple lowest-MCES winner counts =",
    winner_counts
)


pd.DataFrame(
    results
).to_csv(
    OUT,
    index=False
)

print()
print(
    "Written:",
    OUT
)
