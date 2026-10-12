from pathlib import Path
from math import comb

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(r"C:\research\ms_error_geometry")

REAL_REUSED = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_REUSED_FINAL.csv"
)

REAL_NEW_MAP = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_MISSING_FINAL_MAP.csv"
)

REAL_NEW_OUT = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_MISSING_FINAL_OUT.csv"
)

SYN_MAP = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_SYNTHETIC_MAP.csv"
)

SYN_OUT = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_SYNTHETIC_OUT.csv"
)

OUT_LONG = (
    ROOT / "results" /
    "RQ4_CONFIRM_FINAL_LONG_EG358.csv"
)

OUT_PARENT = (
    ROOT / "results" /
    "RQ4_CONFIRM_FINAL_PARENT_MATCHED_EG358.csv"
)

OUT_MODELS = (
    ROOT / "results" /
    "RQ4_CONFIRM_FINAL_MODEL_SENSITIVITY_EG358.csv"
)

OUT_FAMILIES = (
    ROOT / "results" /
    "RQ4_CONFIRM_FINAL_FAMILY_SENSITIVITY_EG358.csv"
)


SEED = 20261012
N_BOOT = 50000

FPGEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def mol(s):
    m = Chem.MolFromSmiles(
        str(s),
        sanitize=False
    )

    if m is None:
        raise RuntimeError(
            f"Invalid SMILES: {s}"
        )

    Chem.SanitizeMol(m)
    Chem.RemoveStereochemistry(m)

    return m


def tani(a, b):
    ma = mol(a)
    mb = mol(b)

    return float(
        DataStructs.TanimotoSimilarity(
            FPGEN.GetFingerprint(ma),
            FPGEN.GetFingerprint(mb),
        )
    )


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
            len(x),
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


def sign_test(x):
    x = np.asarray(
        x,
        dtype=float
    )

    positive = int(
        (x > 0).sum()
    )

    negative = int(
        (x < 0).sum()
    )

    ties = int(
        (x == 0).sum()
    )

    n = positive + negative

    if n == 0:
        return (
            positive,
            negative,
            ties,
            1.0,
        )

    k = min(
        positive,
        negative
    )

    tail = (
        sum(
            comb(n, i)
            for i in range(k + 1)
        )
        /
        (2 ** n)
    )

    p = min(
        1.0,
        2.0 * tail
    )

    return (
        positive,
        negative,
        ties,
        p,
    )


# ============================================================
# REAL: 322 reused + 108 newly computed
# ============================================================

reused = pd.read_csv(
    REAL_REUSED
)

new_map = pd.read_csv(
    REAL_NEW_MAP
)

new_out = pd.read_csv(
    REAL_NEW_OUT,
    header=None,
    names=[
        "pair_id",
        "mces",
        "time_sec",
        "mode",
    ]
)


assert len(reused) == 322
assert len(new_map) == 108
assert len(new_out) == 108

assert not (
    new_out["mces"] < 0
).any()

assert set(
    new_out["mode"]
) == {1}


new = new_map.merge(
    new_out,
    on="pair_id",
    validate="one_to_one",
)

new["mces_source"] = (
    "RQ4_new_exact"
)


# Recalculate Tanimoto for all real observations from the
# authoritative truth representation.
reused["tanimoto"] = [
    tani(a, b)
    for a, b
    in zip(
        reused["truth_smiles"],
        reused["pred_smiles"]
    )
]

new["tanimoto"] = [
    tani(a, b)
    for a, b
    in zip(
        new["truth_smiles"],
        new["pred_smiles"]
    )
]


real_cols = [
    "model",
    "spec",
    "parent_spec",
    "truth_smiles",
    "pred_smiles",
    "tanimoto",
    "mces",
    "mces_source",
]


real = pd.concat(
    [
        reused[real_cols],
        new[real_cols],
    ],
    ignore_index=True,
)


assert len(real) == 430

assert (
    real["parent_spec"].nunique()
    == 333
)

assert not real.duplicated(
    [
        "model",
        "spec",
    ]
).any()


real["source"] = "real"


# ============================================================
# SYNTHETIC: 3261 exact MCES
# ============================================================

syn_map = pd.read_csv(
    SYN_MAP
)

syn_out = pd.read_csv(
    SYN_OUT,
    header=None,
    names=[
        "pair_id",
        "mces",
        "time_sec",
        "mode",
    ]
)


assert len(syn_map) == 3261
assert len(syn_out) == 3261

assert not (
    syn_out["mces"] < 0
).any()

assert set(
    syn_out["mode"]
) == {1}


syn = syn_map.merge(
    syn_out,
    on="pair_id",
    validate="one_to_one",
)


# Recompute for consistency.
syn["tanimoto"] = [
    tani(a, b)
    for a, b
    in zip(
        syn["truth_smiles"],
        syn["pred_smiles"]
    )
]

syn["source"] = "synthetic"


assert (
    syn["parent_spec"].nunique()
    == 333
)


# ============================================================
# Raw descriptive distributions
# ============================================================

print("=" * 110)
print("RQ4 EG358 CONFIRMATORY — FINAL")
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
    real["model"]
    .value_counts()
    .to_string()
)

print()
print(
    "Raw REAL MCES mean/median =",
    real["mces"].mean(),
    "/",
    real["mces"].median()
)

print(
    "Raw SYN MCES mean/median =",
    syn["mces"].mean(),
    "/",
    syn["mces"].median()
)

print()
print(
    "Raw REAL Tanimoto mean/median =",
    real["tanimoto"].mean(),
    "/",
    real["tanimoto"].median()
)

print(
    "Raw SYN Tanimoto mean/median =",
    syn["tanimoto"].mean(),
    "/",
    syn["tanimoto"].median()
)


# ============================================================
# PRIMARY: parent-level matched analysis
# ============================================================

real_parent = (
    real.groupby(
        "parent_spec"
    )
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
            "mces",
            "size"
        ),
    )
    .reset_index()
)


syn_parent = (
    syn.groupby(
        "parent_spec"
    )
    .agg(
        synthetic_median_mces=(
            "mces",
            "median"
        ),

        synthetic_median_tanimoto=(
            "tanimoto",
            "median"
        ),

        n_synthetic=(
            "mces",
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


assert len(matched) == 333


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


# ============================================================
# PRIMARY MCES
# ============================================================

d_mces = matched[
    "delta_mces_real_minus_synthetic"
].to_numpy()


bm = bootstrap(
    d_mces
)

(
    real_farther,
    synthetic_farther,
    mces_ties,
    mces_p,
) = sign_test(
    d_mces
)


sd = d_mces.std(
    ddof=1
)

dz = (
    float(
        d_mces.mean()
        /
        sd
    )
    if sd > 0
    else np.nan
)


print()
print("=" * 110)
print("PRIMARY PARENT-MATCHED MCES")
print("=" * 110)

print(
    "N parents =",
    len(matched)
)

print(
    "REAL median MCES =",
    matched[
        "real_median_mces"
    ].median()
)

print(
    "SYN median MCES =",
    matched[
        "synthetic_median_mces"
    ].median()
)

print(
    "Mean paired delta "
    "(REAL-SYN) =",
    bm["mean"]
)

print(
    "95% CI mean =",
    (
        bm["mean_low"],
        bm["mean_high"],
    )
)

print(
    "Median paired delta =",
    bm["median"]
)

print(
    "95% CI median =",
    (
        bm["median_low"],
        bm["median_high"],
    )
)

print(
    "REAL farther / SYN farther / tie =",
    real_farther,
    "/",
    synthetic_farther,
    "/",
    mces_ties
)

print(
    "Exact sign-test p =",
    mces_p
)

print(
    "Paired dz =",
    dz
)


# ============================================================
# SECONDARY: parent-level Tanimoto
# ============================================================

d_tani = matched[
    "delta_tanimoto_real_minus_synthetic"
].to_numpy()


bt = bootstrap(
    d_tani
)

(
    real_closer,
    synthetic_closer,
    tani_ties,
    tani_p,
) = sign_test(
    d_tani
)


print()
print("=" * 110)
print("SECONDARY PARENT-MATCHED TANIMOTO")
print("=" * 110)

print(
    "REAL median Tanimoto =",
    matched[
        "real_median_tanimoto"
    ].median()
)

print(
    "SYN median Tanimoto =",
    matched[
        "synthetic_median_tanimoto"
    ].median()
)

print(
    "Mean paired delta "
    "(REAL-SYN) =",
    bt["mean"]
)

print(
    "95% CI mean =",
    (
        bt["mean_low"],
        bt["mean_high"],
    )
)

print(
    "Median paired delta =",
    bt["median"]
)

print(
    "REAL closer / SYN closer / tie =",
    real_closer,
    "/",
    synthetic_closer,
    "/",
    tani_ties
)

print(
    "Exact sign-test p =",
    tani_p
)


# ============================================================
# Correlation of the two gap measures
# ============================================================

corr = np.corrcoef(
    matched[
        "delta_mces_real_minus_synthetic"
    ],
    matched[
        "delta_tanimoto_real_minus_synthetic"
    ],
)[0, 1]


print()
print(
    "Correlation MCES gap vs "
    "Tanimoto gap =",
    corr
)


# ============================================================
# MODEL-SPECIFIC SENSITIVITY
# ============================================================

model_rows = []


for model_name in [
    "DiffMS-K1",
    "FRIGID-base@FM1",
    "MS2Forge-K1",
]:

    g = real[
        real["model"]
        == model_name
    ].merge(
        syn_parent[
            [
                "parent_spec",
                "synthetic_median_mces",
                "synthetic_median_tanimoto",
            ]
        ],
        on="parent_spec",
        validate="one_to_one",
    )


    dm = (
        g["mces"]
        -
        g["synthetic_median_mces"]
    ).to_numpy()


    bm2 = bootstrap(
        dm
    )

    (
        rf,
        sf,
        tt,
        pp,
    ) = sign_test(
        dm
    )


    dt = (
        g["tanimoto"]
        -
        g[
            "synthetic_median_tanimoto"
        ]
    ).to_numpy()


    print()
    print("-" * 110)
    print(model_name)
    print("-" * 110)

    print(
        "N =",
        len(g)
    )

    print(
        "REAL median MCES =",
        g["mces"].median()
    )

    print(
        "SYN matched median MCES =",
        g[
            "synthetic_median_mces"
        ].median()
    )

    print(
        "Mean MCES delta =",
        bm2["mean"]
    )

    print(
        "95% CI =",
        (
            bm2["mean_low"],
            bm2["mean_high"],
        )
    )

    print(
        "REAL farther / SYN farther / tie =",
        rf,
        "/",
        sf,
        "/",
        tt
    )

    print(
        "Sign-test p =",
        pp
    )


    model_rows.append({
        "model":
            model_name,

        "n":
            len(g),

        "real_median_mces":
            float(
                g["mces"].median()
            ),

        "synthetic_median_mces":
            float(
                g[
                    "synthetic_median_mces"
                ].median()
            ),

        "mean_mces_delta":
            bm2["mean"],

        "ci_low":
            bm2["mean_low"],

        "ci_high":
            bm2["mean_high"],

        "real_farther":
            rf,

        "synthetic_farther":
            sf,

        "ties":
            tt,

        "sign_p":
            pp,

        "mean_tanimoto_delta":
            float(
                dt.mean()
            ),
    })


models = pd.DataFrame(
    model_rows
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

    sf = (
        syn[
            syn[
                "corruption_family"
            ]
            == family
        ]
        .groupby(
            "parent_spec"
        )
        .agg(
            family_median_mces=(
                "mces",
                "median"
            ),

            family_median_tanimoto=(
                "tanimoto",
                "median"
            ),
        )
        .reset_index()
    )


    q = real_parent.merge(
        sf,
        on="parent_spec",
        how="inner",
        validate="one_to_one",
    )


    dm = (
        q[
            "real_median_mces"
        ]
        -
        q[
            "family_median_mces"
        ]
    ).to_numpy()


    bb = bootstrap(
        dm
    )

    (
        rf,
        sf_count,
        tt,
        pp,
    ) = sign_test(
        dm
    )


    family_rows.append({
        "family":
            family,

        "n_parents":
            len(q),

        "real_median_mces":
            float(
                q[
                    "real_median_mces"
                ].median()
            ),

        "synthetic_median_mces":
            float(
                q[
                    "family_median_mces"
                ].median()
            ),

        "mean_delta":
            bb["mean"],

        "ci_low":
            bb["mean_low"],

        "ci_high":
            bb["mean_high"],

        "real_farther":
            rf,

        "synthetic_farther":
            sf_count,

        "ties":
            tt,

        "sign_p":
            pp,
    })


families = pd.DataFrame(
    family_rows
)


print()
print("=" * 110)
print("CORRUPTION-FAMILY SENSITIVITY")
print("=" * 110)

print(
    families.to_string(
        index=False
    )
)


# ============================================================
# Save publication-ready analysis tables
# ============================================================

real_out = real.copy()

real_out["corruption_family"] = np.nan

syn_out_final = syn[
    [
        "parent_spec",
        "truth_smiles",
        "pred_smiles",
        "tanimoto",
        "mces",
        "corruption_family",
    ]
].copy()

syn_out_final["model"] = np.nan
syn_out_final["spec"] = syn_out_final["parent_spec"]
syn_out_final["mces_source"] = "synthetic_exact"


long = pd.concat(
    [
        real_out[
            [
                "source",
                "model",
                "spec",
                "parent_spec",
                "truth_smiles",
                "pred_smiles",
                "tanimoto",
                "mces",
                "mces_source",
                "corruption_family",
            ]
        ],

        syn_out_final.assign(
            source="synthetic"
        )[
            [
                "source",
                "model",
                "spec",
                "parent_spec",
                "truth_smiles",
                "pred_smiles",
                "tanimoto",
                "mces",
                "mces_source",
                "corruption_family",
            ]
        ],
    ],
    ignore_index=True,
)


long.to_csv(
    OUT_LONG,
    index=False
)

matched.to_csv(
    OUT_PARENT,
    index=False
)

models.to_csv(
    OUT_MODELS,
    index=False
)

families.to_csv(
    OUT_FAMILIES,
    index=False
)


print()
print("=" * 110)
print("FINAL FILES")
print("=" * 110)

print(
    "LONG =",
    OUT_LONG
)

print(
    "PARENT =",
    OUT_PARENT
)

print(
    "MODELS =",
    OUT_MODELS
)

print(
    "FAMILIES =",
    OUT_FAMILIES
)

print()
print(
    "RQ4 EG358 CONFIRMATORY ANALYSIS PASS"
)
