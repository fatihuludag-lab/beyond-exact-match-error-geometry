from pathlib import Path
import pandas as pd

SRC = Path(
    r"results\RQ4_CONFIRM_SYNTHETIC_RAW_EG358.parquet"
)

OUT = Path(
    r"results\RQ4_CONFIRM_SYNTHETIC_PRIMARY_EG358.parquet"
)

d = pd.read_parquet(SRC)

d["parent_spec"] = (
    d["sample_id"]
    .astype(str)
    .str.replace(
        r"__\d+$",
        "",
        regex=True
    )
)

print("=" * 100)
print("RQ4 EG358 SYNTHETIC AUDIT")
print("=" * 100)

print("RAW N =", len(d))
print("RAW PARENTS =", d["parent_spec"].nunique())

print()
print("FAMILIES")
print(
    d["corruption_family"]
    .value_counts()
    .to_string()
)

print()
print(
    "INVERSE VERIFIED =",
    int(d["inverse_verified"].sum()),
    "/",
    len(d)
)

print(
    "INVERSE TRUNCATED =",
    int(
        d[
            "inverse_truncated_same_family"
        ].sum()
    )
)

print(
    "FORWARD TRUNCATED =",
    int(
        d[
            "forward_truncated_family"
        ].sum()
    )
)


primary = d[
    d["inverse_verified"].astype(bool)
    &
    ~d[
        "inverse_truncated_same_family"
    ].astype(bool)
    &
    ~d[
        "forward_truncated_family"
    ].astype(bool)
].copy()


print()
print("=" * 100)
print("PRIMARY SYNTHETIC CORPUS")
print("=" * 100)

print(
    "N =",
    len(primary)
)

print(
    "PARENTS =",
    primary["parent_spec"].nunique()
)

counts = (
    primary.groupby("parent_spec")
    .size()
)

print()
print("PER-PARENT COUNT")
print(
    counts.describe().to_string()
)

print()
print("PRIMARY FAMILIES")
print(
    primary["corruption_family"]
    .value_counts()
    .to_string()
)


# Critical gates
assert len(d) == 3330
assert d["parent_spec"].nunique() == 333

assert (
    primary["parent_spec"].nunique()
    == 333
), "A parent lost all synthetic corruptions."

assert (
    counts.min() > 0
)


primary.to_parquet(
    OUT,
    index=False
)

print()
print(
    "WRITTEN =",
    OUT
)

print()
print(
    "RQ4 EG358 SYNTHETIC AUDIT PASS"
)
