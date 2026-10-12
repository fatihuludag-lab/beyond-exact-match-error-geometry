from pathlib import Path
from math import comb

import numpy as np
import pandas as pd


OLD_REAL_MAP = Path(
    r"results\mces\RQ4_MATCHED_MCES_MAP.csv"
)

OLD_REAL_OUT = Path(
    r"results\mces\RQ4_MATCHED_MCES_OUT.csv"
)

OLD_SYN_MAP = Path(
    r"results\mces\RQ4_SYNTHETIC_ALL860_MAP.csv"
)

OLD_SYN_OUT = Path(
    r"results\mces\RQ4_SYNTHETIC_ALL860_OUT.csv"
)

NEW_MAP = Path(
    r"results\mces\RQ4_V2_INCREMENTAL_41_MAP.csv"
)

NEW_OUT = Path(
    r"results\mces\RQ4_V2_INCREMENTAL_41_OUT.csv"
)

OUT_LONG = Path(
    r"results\RQ4_V2_FINAL_LONG_999.csv"
)

OUT_PARENT = Path(
    r"results\RQ4_V2_FINAL_PARENT_MATCHED_90.csv"
)

OUT_FAMILY = Path(
    r"results\RQ4_V2_FINAL_FAMILY_SENSITIVITY.csv"
)

SEED = 20261008
N_BOOT = 50000


def load_result(map_path, out_path):

    m = pd.read_csv(map_path)

    d = pd.read_csv(
        out_path,
        header=None,
        names=[
            "pair_id",
            "mces",
            "time_sec",
            "mode",
        ]
    )

    if (d["mces"] < 0).any():
        raise RuntimeError(
            f"Failed MCES in {out_path}"
        )

    return m.merge(
        d,
        on="pair_id",
        validate="one_to_one",
    )


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
        / 2**n
    )

    return (
        pos,
        neg,
        tie,
        min(1.0, 2 * tail),
    )


def bootstrap(x):

    x = np.asarray(x, dtype=float)

    rng = np.random.default_rng(SEED)

    idx = rng.integers(
        0,
        len(x),
        size=(N_BOOT, len(x)),
    )

    z = x[idx]

    return {
        "mean": float(x.mean()),
        "mean_low":
            float(np.quantile(z.mean(1), .025)),
        "mean_high":
            float(np.quantile(z.mean(1), .975)),
        "median":
            float(np.median(x)),
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


# ============================================================
# OLD REAL: retain only the 98 real records.
# ============================================================

old_real_all = load_result(
    OLD_REAL_MAP,
    OLD_REAL_OUT
)

old_real = old_real_all[
    old_real_all["source"] == "real"
].copy()

if len(old_real) != 98:
    raise RuntimeError(
        f"Old real != 98: {len(old_real)}"
    )

old_real = old_real[
    [
        "parent_spec",
        "subgroup",
        "tanimoto",
        "mces",
    ]
].copy()

old_real["source"] = "real"


# ============================================================
# OLD SYNTHETIC 860
# ============================================================

old_syn = load_result(
    OLD_SYN_MAP,
    OLD_SYN_OUT
)

if len(old_syn) != 860:
    raise RuntimeError(
        f"Old synthetic != 860: {len(old_syn)}"
    )

old_syn = old_syn[
    [
        "parent_spec",
        "corruption_family",
        "tanimoto",
        "mces",
    ]
].rename(
    columns={
        "corruption_family":
            "subgroup"
    }
)

old_syn["source"] = "synthetic"


# ============================================================
# NEW 41
# ============================================================

new = load_result(
    NEW_MAP,
    NEW_OUT
)

new = new[
    [
        "source",
        "parent_spec",
        "subgroup",
        "tanimoto",
        "mces",
    ]
]


# ============================================================
# FINAL V2 LONG DATA
# ============================================================

long = pd.concat(
    [
        old_real,
        old_syn,
        new,
    ],
    ignore_index=True,
)


real = long[
    long["source"] == "real"
].copy()

syn = long[
    long["source"] == "synthetic"
].copy()


if len(real) != 119:
    raise RuntimeError(
        f"Expected 119 real, got {len(real)}"
    )

if len(syn) != 880:
    raise RuntimeError(
        f"Expected 880 synthetic, got {len(syn)}"
    )

if real["parent_spec"].nunique() != 90:
    raise RuntimeError(
        "Real parent count != 90"
    )

if syn["parent_spec"].nunique() != 90:
    raise RuntimeError(
        "Synthetic parent count != 90"
    )


print("=" * 110)
print("RQ4 V2 FINAL — THREE GENERATORS")
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
    "PARENTS =",
    real["parent_spec"].nunique()
)

print()
print("REAL MODEL COUNTS")

print(
    real["subgroup"]
    .value_counts()
    .to_string()
)

print()
print(
    "Raw real MCES mean/median =",
    real["mces"].mean(),
    "/",
    real["mces"].median()
)

print(
    "Raw synthetic MCES mean/median =",
    syn["mces"].mean(),
    "/",
    syn["mces"].median()
)


# ============================================================
# Parent-level matched analysis
# ============================================================

real_parent = (
    real.groupby("parent_spec")
    .agg(
        real_median_mces=("mces", "median"),
        real_median_tanimoto=("tanimoto", "median"),
        n_real_errors=("mces", "size"),
    )
    .reset_index()
)

syn_parent = (
    syn.groupby("parent_spec")
    .agg(
        synthetic_median_mces=("mces", "median"),
        synthetic_median_tanimoto=("tanimoto", "median"),
        n_synthetic=("mces", "size"),
    )
    .reset_index()
)

matched = real_parent.merge(
    syn_parent,
    on="parent_spec",
    validate="one_to_one",
)

if len(matched) != 90:
    raise RuntimeError(
        f"Matched parent count !=90: {len(matched)}"
    )


matched["delta_mces_real_minus_synthetic"] = (
    matched["real_median_mces"]
    -
    matched["synthetic_median_mces"]
)

matched["delta_tanimoto_real_minus_synthetic"] = (
    matched["real_median_tanimoto"]
    -
    matched["synthetic_median_tanimoto"]
)


d_mces = matched[
    "delta_mces_real_minus_synthetic"
].to_numpy()

b = bootstrap(d_mces)

pos, neg, ties, p = sign_test(d_mces)


print()
print("=" * 110)
print("PARENT-MATCHED MCES")
print("=" * 110)

print(
    "Real median MCES =",
    matched[
        "real_median_mces"
    ].median()
)

print(
    "Synthetic median MCES =",
    matched[
        "synthetic_median_mces"
    ].median()
)

print(
    "Mean delta =",
    b["mean"]
)

print(
    "95% CI mean =",
    (
        b["mean_low"],
        b["mean_high"]
    )
)

print(
    "Median delta =",
    b["median"]
)

print(
    "95% CI median =",
    (
        b["median_low"],
        b["median_high"]
    )
)

print(
    "REAL farther / SYN farther / tie =",
    pos,
    "/",
    neg,
    "/",
    ties
)

print(
    "Sign-test p =",
    p
)


sd = d_mces.std(ddof=1)

print(
    "Paired dz =",
    d_mces.mean() / sd
)


# ============================================================
# Tanimoto confirmation
# ============================================================

d_tani = matched[
    "delta_tanimoto_real_minus_synthetic"
].to_numpy()

bt = bootstrap(d_tani)

po, ne, ti, pv = sign_test(d_tani)


print()
print("=" * 110)
print("PARENT-MATCHED TANIMOTO")
print("=" * 110)

print(
    "Real median Tanimoto =",
    matched[
        "real_median_tanimoto"
    ].median()
)

print(
    "Synthetic median Tanimoto =",
    matched[
        "synthetic_median_tanimoto"
    ].median()
)

print(
    "Mean real-synthetic delta =",
    bt["mean"]
)

print(
    "95% CI =",
    (
        bt["mean_low"],
        bt["mean_high"]
    )
)

print(
    "REAL closer / SYN closer / tie =",
    po,
    "/",
    ne,
    "/",
    ti
)

print(
    "Sign-test p =",
    pv
)


# ============================================================
# Model-specific sensitivity
# ============================================================

for model in [
    "DiffMS-K1",
    "FRIGID-base@FM1",
    "MS2Forge-K1",
]:

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
        g["synthetic_median_mces"]
    ).to_numpy()

    bb = bootstrap(d)
    pp, nn, tt, pv2 = sign_test(d)

    print()
    print("-" * 110)
    print(model)
    print("-" * 110)

    print("N =", len(g))

    print(
        "Real MCES median =",
        g["mces"].median()
    )

    print(
        "Synthetic matched median =",
        g[
            "synthetic_median_mces"
        ].median()
    )

    print(
        "Mean delta =",
        bb["mean"]
    )

    print(
        "95% CI =",
        (
            bb["mean_low"],
            bb["mean_high"]
        )
    )

    print(
        "real farther / synthetic farther / tie =",
        pp, "/", nn, "/", tt
    )

    print(
        "sign p =",
        pv2
    )


# ============================================================
# Family sensitivity
# ============================================================

family_rows = []

for family in sorted(
    syn["subgroup"].unique()
):

    sf = (
        syn[
            syn["subgroup"] == family
        ]
        .groupby("parent_spec")
        .agg(
            family_median_mces=("mces", "median")
        )
        .reset_index()
    )

    q = real_parent.merge(
        sf,
        on="parent_spec",
        how="inner",
        validate="one_to_one",
    )

    d = (
        q["real_median_mces"]
        -
        q["family_median_mces"]
    ).to_numpy()

    bb = bootstrap(d)
    pp, nn, tt, pv2 = sign_test(d)

    family_rows.append({
        "family": family,
        "n_parents": len(q),
        "real_median_mces":
            q["real_median_mces"].median(),
        "synthetic_median_mces":
            q["family_median_mces"].median(),
        "mean_delta": bb["mean"],
        "ci_low": bb["mean_low"],
        "ci_high": bb["mean_high"],
        "real_farther": pp,
        "synthetic_farther": nn,
        "ties": tt,
        "sign_p": pv2,
    })


family = pd.DataFrame(
    family_rows
)

print()
print("=" * 110)
print("FAMILY SENSITIVITY")
print("=" * 110)

print(
    family.to_string(
        index=False
    )
)


OUT_LONG.parent.mkdir(
    parents=True,
    exist_ok=True
)

long.to_csv(
    OUT_LONG,
    index=False
)

matched.to_csv(
    OUT_PARENT,
    index=False
)

family.to_csv(
    OUT_FAMILY,
    index=False
)

print()
print("LONG   =", OUT_LONG)
print("PARENT =", OUT_PARENT)
print("FAMILY =", OUT_FAMILY)
