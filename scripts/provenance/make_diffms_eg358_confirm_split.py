from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\research\ms_error_geometry")

PANEL = (
    ROOT / "manifests" /
    "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv"
)

ORIGINAL_SPLIT = Path(
    r"C:\research\DiffMS\data\msg\split.tsv"
)

OUT = (
    ROOT / "manifests" /
    "DIFFMS_MSG_EG358_CONFIRM_split.tsv"
)

panel = pd.read_csv(PANEL)
split = pd.read_csv(ORIGINAL_SPLIT, sep="\t")

print("ORIGINAL COLUMNS =", split.columns.tolist())
print("ORIGINAL N =", len(split))
print()
print("ORIGINAL SPLIT COUNTS")
print(split["split"].value_counts().to_string())

# DiffMS/MassSpecGym versions may use name or spec.
if "name" in split.columns:
    id_col = "name"
elif "spec" in split.columns:
    id_col = "spec"
else:
    raise RuntimeError(
        f"Cannot identify spectrum ID column: {split.columns.tolist()}"
    )

selected = set(
    panel["spec"].astype(str)
)

available = set(
    split[id_col].astype(str)
)

missing = selected - available

if missing:
    raise RuntimeError(
        f"{len(missing)} EG358 spectra absent from original split."
    )

# Confirm all selected spectra belonged to original test split.
original_selected = split[
    split[id_col].astype(str).isin(selected)
]

print()
print(
    "SELECTED ORIGINAL SPLITS =",
    original_selected["split"]
    .value_counts()
    .to_dict()
)

if len(original_selected) != 358:
    raise RuntimeError(
        f"Expected 358 selected spectra, got {len(original_selected)}"
    )

if not (
    original_selected["split"] == "test"
).all():
    raise RuntimeError(
        "At least one confirmatory spectrum was not originally test."
    )


out = split.copy()

# All former test spectra not selected for EG358 are moved out
# of the evaluation split. Existing train/val entries remain unchanged.
old_test = (
    out["split"] == "test"
)

out.loc[
    old_test
    & ~out[id_col].astype(str).isin(selected),
    "split"
] = "val"

out.loc[
    out[id_col].astype(str).isin(selected),
    "split"
] = "test"


test_ids = set(
    out.loc[
        out["split"] == "test",
        id_col
    ].astype(str)
)

assert test_ids == selected
assert len(test_ids) == 358

out.to_csv(
    OUT,
    sep="\t",
    index=False,
    lineterminator="\n"
)

print()
print("=" * 90)
print("DIFFMS EG358 CONFIRM SPLIT")
print("=" * 90)

print(
    "TEST N =",
    int((out["split"] == "test").sum())
)

print(
    "TEST == EG358 =",
    test_ids == selected
)

print()
print(
    out["split"]
    .value_counts()
    .to_string()
)

print()
print("WRITTEN =", OUT)
print("DIFFMS EG358 SPLIT PASS")
