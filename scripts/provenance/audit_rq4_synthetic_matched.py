from pathlib import Path
import pandas as pd


SRC = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100.parquet"
)

OUT_ALL = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_AUDITED.parquet"
)

OUT_PRIMARY = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_PRIMARY.parquet"
)

SUMMARY = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_SUMMARY.csv"
)


d = pd.read_parquet(SRC)


# Generator appends __00, __01, ...
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
print("RQ4 SYNTHETIC MATCHED CORPUS AUDIT")
print("=" * 100)

print("ROWS =", len(d))
print(
    "UNIQUE PARENTS =",
    d["parent_spec"].nunique()
)

counts = (
    d.groupby("parent_spec")
    .size()
)

print()
print("CORRUPTIONS PER PARENT")
print(counts.describe().to_string())

print()
print(
    "PARENTS WITH EXACTLY 10 =",
    int((counts == 10).sum()),
    "/",
    len(counts)
)

print()
print("FAMILY COUNTS")
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


# ------------------------------------------------------------
# Primary RQ4 synthetic set:
#
# - inverse verified
# - no inverse truncation
# - no forward truncation
#
# Full 880 set will remain available as sensitivity analysis.
# ------------------------------------------------------------

primary = d[
    d["inverse_verified"]
    &
    ~d["inverse_truncated_same_family"]
    &
    ~d["forward_truncated_family"]
].copy()


print()
print("=" * 100)
print("PRIMARY NON-TRUNCATED SET")
print("=" * 100)

print("N =", len(primary))

print(
    "UNIQUE PARENTS =",
    primary[
        "parent_spec"
    ].nunique()
)

primary_counts = (
    primary.groupby("parent_spec")
    .size()
)

print()
print(
    primary_counts
    .describe()
    .to_string()
)


# ------------------------------------------------------------
# Parent-level summary
# ------------------------------------------------------------

summary = (
    d.groupby("parent_spec")
    .agg(
        n_total=(
            "sample_id",
            "size"
        ),

        n_forward_truncated=(
            "forward_truncated_family",
            "sum"
        ),

        n_inverse_truncated=(
            "inverse_truncated_same_family",
            "sum"
        ),

        n_verified=(
            "inverse_verified",
            "sum"
        ),
    )
    .reset_index()
)

summary[
    "n_primary"
] = (
    summary["n_total"]
    -
    summary["n_forward_truncated"]
    -
    summary["n_inverse_truncated"]
)


OUT_ALL.parent.mkdir(
    parents=True,
    exist_ok=True
)

d.to_parquet(
    OUT_ALL,
    index=False
)

primary.to_parquet(
    OUT_PRIMARY,
    index=False
)

summary.to_csv(
    SUMMARY,
    index=False
)

print()
print("ALL     =", OUT_ALL)
print("PRIMARY =", OUT_PRIMARY)
print("SUMMARY =", SUMMARY)

print()
print("RQ4 SYNTHETIC AUDIT PASS")
