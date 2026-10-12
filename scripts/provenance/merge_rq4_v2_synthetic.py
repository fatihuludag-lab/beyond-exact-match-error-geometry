from pathlib import Path
import pandas as pd

OLD = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_PRIMARY.parquet"
)

NEW = Path(
    r"results\RQ4_V2_SYNTHETIC_NEW2_RAW.parquet"
)

OUT = Path(
    r"results\RQ4_V2_SYNTHETIC_PRIMARY_90PARENTS.parquet"
)

old = pd.read_parquet(OLD)
new = pd.read_parquet(NEW)

if "parent_spec" not in old.columns:
    old["parent_spec"] = (
        old["sample_id"]
        .astype(str)
        .str.replace(r"__\d+$", "", regex=True)
    )

new["parent_spec"] = (
    new["sample_id"]
    .astype(str)
    .str.replace(r"__\d+$", "", regex=True)
)

new_primary = new[
    new["inverse_verified"]
    &
    ~new["inverse_truncated_same_family"]
    &
    ~new["forward_truncated_family"]
].copy()

print("=" * 90)
print("NEW TWO-PARENT SYNTHETIC")
print("=" * 90)

print("raw =", len(new))
print("primary =", len(new_primary))
print("parents =", new_primary["parent_spec"].nunique())

combined = pd.concat(
    [old, new_primary],
    ignore_index=True,
)

if combined["sample_id"].duplicated().any():
    raise RuntimeError("Duplicate synthetic sample_id")

print()
print("=" * 90)
print("RQ4 V2 COMBINED SYNTHETIC")
print("=" * 90)

print("N =", len(combined))
print(
    "PARENTS =",
    combined["parent_spec"].nunique()
)

print()
print(
    combined.groupby("parent_spec")
    .size()
    .describe()
    .to_string()
)

print()
print("FAMILIES")
print(
    combined["corruption_family"]
    .value_counts()
    .to_string()
)

if combined["parent_spec"].nunique() != 90:
    raise RuntimeError(
        "Expected exactly 90 parents"
    )

combined.to_parquet(
    OUT,
    index=False
)

print()
print("Written:", OUT)
print("RQ4 V2 SYNTHETIC MERGE PASS")
