from pathlib import Path
from math import comb
import numpy as np
import pandas as pd

FILES = {
    "DiffMS": Path(r"results\normalized\DIFFMS_K1_EG358_CONFIRM.csv"),
    "FRIGID": Path(r"results\normalized\FRIGID_BASE_FM1_EG358_CONFIRM_HARMONIZED.csv"),
    "MS2Forge": Path(r"results\normalized\MS2FORGE_K1_EG358_CONFIRM.csv"),
}

OUT = Path(r"results\EG358_CONFIRM_THREE_MODEL_PAIRED.csv")

SEED = 20261011
NBOOT = 50000


def load(name, path):
    d = pd.read_csv(path)

    return d[[
        "spec",
        "task_valid",
        "same_formula",
        "exact_connectivity",
        "tanimoto",
        "truth_smiles",
        "pred_smiles",
    ]].rename(columns={
        "task_valid": f"{name}_valid",
        "same_formula": f"{name}_formula",
        "exact_connectivity": f"{name}_exact",
        "tanimoto": f"{name}_tanimoto",
        "pred_smiles": f"{name}_pred",
    })


def sign_test(x):
    x = np.asarray(x, float)

    pos = int((x > 0).sum())
    neg = int((x < 0).sum())
    ties = int((x == 0).sum())

    n = pos + neg

    if n == 0:
        return pos, neg, ties, 1.0

    k = min(pos, neg)

    p = 2 * sum(
        comb(n, i)
        for i in range(k + 1)
    ) / (2 ** n)

    return pos, neg, ties, min(1.0, p)


def boot_mean(x):
    x = np.asarray(x, float)

    rng = np.random.default_rng(SEED)

    idx = rng.integers(
        0, len(x),
        size=(NBOOT, len(x))
    )

    vals = x[idx].mean(axis=1)

    return (
        float(x.mean()),
        float(np.quantile(vals, .025)),
        float(np.quantile(vals, .975)),
    )


x = load("DiffMS", FILES["DiffMS"])

for name in ["FRIGID", "MS2Forge"]:
    y = load(name, FILES[name])

    # truth_smiles only once
    y = y.drop(columns=["truth_smiles"])

    x = x.merge(
        y,
        on="spec",
        validate="one_to_one"
    )

assert len(x) == 358


print("=" * 100)
print("EG358 CONFIRMATORY — THREE MODEL SUMMARY")
print("=" * 100)

for m in FILES:
    print(
        m,
        "valid=", int(x[f"{m}_valid"].sum()),
        "formula=", int(x[f"{m}_formula"].sum()),
        "exact=", int(x[f"{m}_exact"].sum())
    )


pairs = [
    ("DiffMS", "FRIGID"),
    ("DiffMS", "MS2Forge"),
    ("FRIGID", "MS2Forge"),
]


for A, B in pairs:

    q = x[
        x[f"{A}_valid"]
        &
        x[f"{B}_valid"]
        &
        ~x[f"{A}_exact"]
        &
        ~x[f"{B}_exact"]
    ].copy()

    delta = (
        q[f"{B}_tanimoto"]
        -
        q[f"{A}_tanimoto"]
    ).to_numpy()

    mean, lo, hi = boot_mean(delta)

    pos, neg, ties, p = sign_test(delta)

    print()
    print("=" * 100)
    print(f"{A} vs {B} — COMMON VALID WRONG")
    print("=" * 100)

    print("N =", len(q))

    print(
        A,
        "mean/median =",
        q[f"{A}_tanimoto"].mean(),
        "/",
        q[f"{A}_tanimoto"].median()
    )

    print(
        B,
        "mean/median =",
        q[f"{B}_tanimoto"].mean(),
        "/",
        q[f"{B}_tanimoto"].median()
    )

    print(
        f"Mean delta {B}-{A} =",
        mean
    )

    print(
        "95% CI =",
        (lo, hi)
    )

    print(
        B, "better =", pos,
        A, "better =", neg,
        "ties =", ties
    )

    print(
        "Sign-test p =",
        p
    )

    both_formula = q[
        q[f"{A}_formula"]
        &
        q[f"{B}_formula"]
    ]

    print(
        "BOTH-SAME-FORMULA N =",
        len(both_formula)
    )


# Triple subset
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
print("=" * 100)
print("TRIPLE COMMON VALID WRONG")
print("=" * 100)

print("N =", len(triple))

for m in FILES:
    print(
        m,
        "mean/median =",
        triple[f"{m}_tanimoto"].mean(),
        "/",
        triple[f"{m}_tanimoto"].median()
    )


x.to_csv(
    OUT,
    index=False
)

print()
print("WRITTEN =", OUT)
