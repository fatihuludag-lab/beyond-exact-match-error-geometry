from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


MAP = Path(
    r"results\mces\EG358_CONFIRM_RQ2_MCES_MAP.csv"
)

MCES = Path(
    r"results\mces\EG358_CONFIRM_RQ2_MCES_OUT.csv"
)

COMP = Path(
    r"results\mces\EG358_CONFIRM_RQ2_COMPARISONS.csv"
)

OUT = Path(
    r"results\EG358_CONFIRM_RQ2_MCES_FINAL.csv"
)

SEED = 20261012
N_BOOT = 50000


def bootstrap(x):
    x = np.asarray(x, dtype=float)

    rng = np.random.default_rng(SEED)

    idx = rng.integers(
        0,
        len(x),
        size=(N_BOOT, len(x)),
    )

    z = x[idx]

    means = z.mean(axis=1)
    medians = np.median(z, axis=1)

    return {
        "mean": float(x.mean()),
        "mean_low": float(np.quantile(means, 0.025)),
        "mean_high": float(np.quantile(means, 0.975)),
        "median": float(np.median(x)),
        "median_low": float(np.quantile(medians, 0.025)),
        "median_high": float(np.quantile(medians, 0.975)),
    }


def sign_test_mces(delta):
    """
    delta = B - A

    MCES lower is better:
      delta < 0 => B better
      delta > 0 => A better
    """

    x = np.asarray(delta, dtype=float)

    b_better = int((x < 0).sum())
    a_better = int((x > 0).sum())
    ties = int((x == 0).sum())

    n = b_better + a_better

    if n == 0:
        return b_better, a_better, ties, 1.0

    k = min(
        b_better,
        a_better,
    )

    tail = (
        sum(
            comb(n, i)
            for i in range(k + 1)
        )
        / 2**n
    )

    return (
        b_better,
        a_better,
        ties,
        min(1.0, 2 * tail),
    )


mapping = pd.read_csv(MAP)

dist = pd.read_csv(
    MCES,
    header=None,
    names=[
        "pair_id",
        "mces",
        "time_sec",
        "mode",
    ],
)

comp = pd.read_csv(COMP)


# ------------------------------------------------------------
# Hard gates
# ------------------------------------------------------------

assert len(mapping) == 585
assert len(dist) == 585
assert dist["pair_id"].nunique() == 585
assert not (dist["mces"] < 0).any()
assert set(dist["mode"]) == {1}


long = mapping.merge(
    dist,
    on="pair_id",
    validate="one_to_one",
)


print("=" * 110)
print("EG358 CONFIRMATORY RQ2 — EXACT MCES")
print("=" * 110)

print(
    "MCES pairs =",
    len(long)
)

print(
    "Exact mode =",
    int((long["mode"] == 1).sum()),
    "/",
    len(long)
)

print(
    "Median compute time =",
    long["time_sec"].median()
)

print(
    "Max compute time =",
    long["time_sec"].max()
)


comparison_order = [
    "DiffMS_vs_FRIGID",
    "DiffMS_vs_MS2Forge",
    "FRIGID_vs_MS2Forge",
]

summary_rows = []


# ============================================================
# Pairwise confirmatory analyses
# ============================================================

for cname in comparison_order:

    c = comp[
        comp["comparison"] == cname
    ].copy()

    A = c["model_A"].iloc[0]
    B = c["model_B"].iloc[0]

    specs = set(c["spec"])

    q = long[
        long["spec"].isin(specs)
        &
        long["model"].isin([A, B])
    ].copy()


    mces = (
        q.pivot(
            index="spec",
            columns="model",
            values="mces",
        )
        .reset_index()
    )

    formula = (
        q.pivot(
            index="spec",
            columns="model",
            values="same_formula",
        )
        .reset_index()
    )

    tani = (
        q.pivot(
            index="spec",
            columns="model",
            values="tanimoto",
        )
        .reset_index()
    )


    if len(mces) != len(c):
        raise RuntimeError(
            f"{cname}: expected {len(c)}, got {len(mces)}"
        )

    if mces[[A, B]].isna().any().any():
        raise RuntimeError(
            f"{cname}: missing MCES value"
        )


    delta = (
        mces[B] - mces[A]
    ).to_numpy(dtype=float)

    boot = bootstrap(delta)

    (
        b_better,
        a_better,
        ties,
        p,
    ) = sign_test_mces(delta)


    sd = delta.std(ddof=1)

    dz = (
        float(delta.mean() / sd)
        if sd > 0
        else np.nan
    )


    print()
    print("=" * 110)
    print(
        f"{A} vs {B} — COMMON VALID WRONG"
    )
    print("=" * 110)

    print(
        "N =",
        len(mces)
    )

    print(
        f"{A} MCES mean/median =",
        mces[A].mean(),
        "/",
        mces[A].median()
    )

    print(
        f"{B} MCES mean/median =",
        mces[B].mean(),
        "/",
        mces[B].median()
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
        b_better
    )

    print(
        f"{A} better =",
        a_better
    )

    print(
        "Ties =",
        ties
    )

    print(
        "Exact sign-test p =",
        p
    )

    print(
        "Paired dz =",
        dz
    )


    # --------------------------------------------------------
    # BOTH SAME FORMULA sensitivity
    # --------------------------------------------------------

    sf_specs = formula.loc[
        formula[A].astype(bool)
        &
        formula[B].astype(bool),
        "spec",
    ]

    sf = mces[
        mces["spec"].isin(sf_specs)
    ].copy()

    print()
    print(
        "BOTH-SAME-FORMULA N =",
        len(sf)
    )

    sf_result = {
        "sf_n": len(sf),
        "sf_mean_delta": np.nan,
        "sf_ci_low": np.nan,
        "sf_ci_high": np.nan,
        "sf_B_better": np.nan,
        "sf_A_better": np.nan,
        "sf_ties": np.nan,
        "sf_p": np.nan,
    }

    if len(sf):

        d_sf = (
            sf[B] - sf[A]
        ).to_numpy(dtype=float)

        bs = bootstrap(d_sf)

        (
            bb,
            aa,
            tt,
            pp,
        ) = sign_test_mces(d_sf)

        print(
            f"{A} MCES median =",
            sf[A].median()
        )

        print(
            f"{B} MCES median =",
            sf[B].median()
        )

        print(
            f"Mean delta {B}-{A} =",
            bs["mean"]
        )

        print(
            "95% CI mean =",
            (
                bs["mean_low"],
                bs["mean_high"],
            )
        )

        print(
            f"{B} better / {A} better / tie =",
            bb, "/", aa, "/", tt
        )

        print(
            "Sign-test p =",
            pp
        )

        sf_result = {
            "sf_n": len(sf),
            "sf_mean_delta": bs["mean"],
            "sf_ci_low": bs["mean_low"],
            "sf_ci_high": bs["mean_high"],
            "sf_B_better": bb,
            "sf_A_better": aa,
            "sf_ties": tt,
            "sf_p": pp,
        }


    # descriptive Tanimoto on exactly same pairwise population
    tani_pair = tani[
        tani["spec"].isin(
            mces["spec"]
        )
    ]

    summary_rows.append({
        "comparison":
            cname,

        "model_A":
            A,

        "model_B":
            B,

        "n":
            len(mces),

        "A_mean_mces":
            float(mces[A].mean()),

        "A_median_mces":
            float(mces[A].median()),

        "B_mean_mces":
            float(mces[B].mean()),

        "B_median_mces":
            float(mces[B].median()),

        "mean_delta_B_minus_A":
            boot["mean"],

        "mean_ci_low":
            boot["mean_low"],

        "mean_ci_high":
            boot["mean_high"],

        "median_delta_B_minus_A":
            boot["median"],

        "median_ci_low":
            boot["median_low"],

        "median_ci_high":
            boot["median_high"],

        "B_better":
            b_better,

        "A_better":
            a_better,

        "ties":
            ties,

        "sign_test_p":
            p,

        "paired_dz":
            dz,

        "A_mean_tanimoto":
            float(tani_pair[A].mean()),

        "B_mean_tanimoto":
            float(tani_pair[B].mean()),

        **sf_result,
    })


# ============================================================
# Triple common-valid-wrong population
# ============================================================

sets = {}

for cname in comparison_order:
    sets[cname] = set(
        comp.loc[
            comp["comparison"] == cname,
            "spec"
        ]
    )

triple_specs = (
    sets["DiffMS_vs_FRIGID"]
    &
    sets["DiffMS_vs_MS2Forge"]
    &
    sets["FRIGID_vs_MS2Forge"]
)

assert len(triple_specs) == 93


triple_long = long[
    long["spec"].isin(triple_specs)
].copy()


triple = (
    triple_long.pivot(
        index="spec",
        columns="model",
        values="mces",
    )
    .reset_index()
)

assert len(triple) == 93


print()
print("=" * 110)
print("TRIPLE COMMON VALID WRONG — MCES")
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


winner_counts = {
    "DiffMS": 0,
    "FRIGID": 0,
    "MS2Forge": 0,
    "tie": 0,
}


for r in triple.itertuples(index=False):

    values = {
        "DiffMS": r.DiffMS,
        "FRIGID": r.FRIGID,
        "MS2Forge": r.MS2Forge,
    }

    best = min(values.values())

    winners = [
        name
        for name, value
        in values.items()
        if value == best
    ]

    if len(winners) == 1:
        winner_counts[winners[0]] += 1
    else:
        winner_counts["tie"] += 1


print()
print(
    "Lowest-MCES winner counts =",
    winner_counts
)


# ============================================================
# Save confirmatory result
# ============================================================

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUT,
    index=False,
)

print()
print("=" * 110)
print("FINAL CONFIRMATORY TABLE")
print("=" * 110)

print(
    summary[
        [
            "comparison",
            "n",
            "A_mean_mces",
            "B_mean_mces",
            "mean_delta_B_minus_A",
            "mean_ci_low",
            "mean_ci_high",
            "B_better",
            "A_better",
            "ties",
            "sign_test_p",
            "sf_n",
        ]
    ].to_string(index=False)
)

print()
print(
    "WRITTEN =",
    OUT
)

print()
print(
    "EG358 CONFIRMATORY RQ2 MCES ANALYSIS PASS"
)
