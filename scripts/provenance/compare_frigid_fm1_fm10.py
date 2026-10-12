from pathlib import Path
import pandas as pd

A = Path(
    r"results\normalized\FRIGID_BASE_EG100_SMOKE10.csv"
)

B = Path(
    r"results\normalized\FRIGID_BASE_FM10_SMOKE10.csv"
)

OUT = Path(
    r"results\frigid_fm1_vs_fm10_smoke10_paired.csv"
)

a = pd.read_csv(A)
b = pd.read_csv(B)

keep = [
    "spec",
    "valid",
    "same_formula",
    "exact_connectivity",
    "tanimoto",
]

a = a[keep].rename(
    columns={
        c: f"fm1_{c}"
        for c in keep
        if c != "spec"
    }
)

b = b[keep].rename(
    columns={
        c: f"fm10_{c}"
        for c in keep
        if c != "spec"
    }
)

d = a.merge(
    b,
    on="spec",
    how="inner",
    validate="one_to_one",
)

d["delta_tanimoto"] = (
    d["fm10_tanimoto"]
    - d["fm1_tanimoto"]
)

d["exact_transition"] = (
    d["fm1_exact_connectivity"]
    .astype(int)
    .astype(str)
    + "->"
    + d["fm10_exact_connectivity"]
    .astype(int)
    .astype(str)
)

common_wrong = d[
    (~d["fm1_exact_connectivity"])
    &
    (~d["fm10_exact_connectivity"])
].copy()

print("=" * 100)
print("FRIGID FM1 vs FM10 — PAIRED SMOKE AUDIT")
print("=" * 100)

print(
    d[
        [
            "spec",
            "fm1_exact_connectivity",
            "fm10_exact_connectivity",
            "fm1_same_formula",
            "fm10_same_formula",
            "fm1_tanimoto",
            "fm10_tanimoto",
            "delta_tanimoto",
        ]
    ].to_string(index=False)
)

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    "FM1 exact =",
    int(d["fm1_exact_connectivity"].sum()),
    "/",
    len(d),
)

print(
    "FM10 exact =",
    int(d["fm10_exact_connectivity"].sum()),
    "/",
    len(d),
)

print(
    "FM1 formula =",
    int(d["fm1_same_formula"].sum()),
    "/",
    len(d),
)

print(
    "FM10 formula =",
    int(d["fm10_same_formula"].sum()),
    "/",
    len(d),
)

print()
print(
    "Mean FM1 Tanimoto =",
    d["fm1_tanimoto"].mean()
)

print(
    "Mean FM10 Tanimoto =",
    d["fm10_tanimoto"].mean()
)

print(
    "Mean delta =",
    d["delta_tanimoto"].mean()
)

print(
    "Median delta =",
    d["delta_tanimoto"].median()
)

print(
    "FM10 improved Tanimoto =",
    int((d["delta_tanimoto"] > 0).sum()),
    "/",
    len(d),
)

print(
    "FM10 worsened Tanimoto =",
    int((d["delta_tanimoto"] < 0).sum()),
    "/",
    len(d),
)

print()
print("EXACT TRANSITIONS")
print(
    d["exact_transition"]
    .value_counts()
    .sort_index()
    .to_string()
)

print()
print(
    "COMMON WRONG =",
    len(common_wrong)
)

if len(common_wrong):

    print(
        "Common-wrong FM1 mean Tanimoto =",
        common_wrong[
            "fm1_tanimoto"
        ].mean()
    )

    print(
        "Common-wrong FM10 mean Tanimoto =",
        common_wrong[
            "fm10_tanimoto"
        ].mean()
    )

    print(
        "Common-wrong mean delta =",
        common_wrong[
            "delta_tanimoto"
        ].mean()
    )

    print(
        "Common-wrong FM10 better =",
        int(
            (
                common_wrong[
                    "delta_tanimoto"
                ] > 0
            ).sum()
        ),
        "/",
        len(common_wrong),
    )

d.to_csv(
    OUT,
    index=False
)

print()
print("Written:", OUT)
