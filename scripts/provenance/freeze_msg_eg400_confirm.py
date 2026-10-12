from pathlib import Path
import hashlib
import json
import shutil

import pandas as pd


ROOT = Path(r"C:\research\ms_error_geometry")

EG100 = ROOT / "manifests" / "MSG_EG100_DEV_v01.csv"
EG500 = ROOT / "manifests" / "MSG_EG500_PREFREEZE_v01.csv"

SOURCE = Path(r"C:\research\DiffMS\data\msg")

OUT_MANIFEST = (
    ROOT / "manifests" /
    "MSG_EG400_CONFIRM_v01.csv"
)

OUT_DATA = (
    ROOT / "data" /
    "MSG_EG400_CONFIRM_v01"
)

META = (
    ROOT / "manifests" /
    "MSG_EG400_CONFIRM_FREEZE_v01.json"
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


e100 = pd.read_csv(EG100)
e500 = pd.read_csv(EG500)

dev = set(e100["connectivity_key"])

confirm = e500[
    ~e500["connectivity_key"].isin(dev)
].copy()

confirm = (
    confirm
    .sort_values("panel_index")
    .reset_index(drop=True)
)

confirm["confirm_index"] = range(
    1,
    len(confirm) + 1
)


assert len(e100) == 100
assert len(e500) == 500
assert len(confirm) == 400

assert not (
    set(confirm["connectivity_key"])
    &
    set(e100["connectivity_key"])
)

assert (
    set(confirm["connectivity_key"])
    |
    set(e100["connectivity_key"])
) == set(e500["connectivity_key"])


confirm.to_csv(
    OUT_MANIFEST,
    index=False,
    lineterminator="\n"
)


# ---------------------------------------------------------
# Minimal MassSpecGym directory
# ---------------------------------------------------------

if OUT_DATA.exists():
    shutil.rmtree(OUT_DATA)

spec_out = OUT_DATA / "spec_files"

sub_out = (
    OUT_DATA /
    "subformulae" /
    "default_subformulae"
)

spec_out.mkdir(
    parents=True
)

sub_out.mkdir(
    parents=True
)


labels = pd.read_csv(
    SOURCE / "labels.tsv",
    sep="\t"
)

selected = set(confirm["spec"])

labels_sub = labels[
    labels["spec"].isin(selected)
].copy()

order = {
    spec: i
    for i, spec
    in enumerate(confirm["spec"])
}

labels_sub["_order"] = (
    labels_sub["spec"]
    .map(order)
)

labels_sub = (
    labels_sub
    .sort_values("_order")
    .drop(columns="_order")
)

labels_sub.to_csv(
    OUT_DATA / "labels.tsv",
    sep="\t",
    index=False,
    lineterminator="\n"
)

pd.DataFrame({
    "name": confirm["spec"],
    "split": ["test"] * 400,
}).to_csv(
    OUT_DATA / "split.tsv",
    sep="\t",
    index=False,
    lineterminator="\n"
)


for spec in confirm["spec"]:

    shutil.copy2(
        SOURCE /
        "spec_files" /
        f"{spec}.ms",

        spec_out /
        f"{spec}.ms",
    )

    shutil.copy2(
        SOURCE /
        "subformulae" /
        "default_subformulae" /
        f"{spec}.json",

        sub_out /
        f"{spec}.json",
    )


metadata = {
    "panel":
        "MSG-EG400-CONFIRM-v01",

    "status":
        "FROZEN_CONFIRMATORY",

    "parent_panel":
        "MSG-EG500-PREFREEZE-v01",

    "excluded_development_panel":
        "MSG-EG100-DEV-v01",

    "n":
        400,

    "overlap_with_EG100":
        0,

    "manifest_sha256":
        sha256(OUT_MANIFEST),

    "primary_endpoints": {
        "RQ2":
            "paired MCES difference on common-valid-wrong cases",

        "RQ4":
            "parent-level median real MCES minus median synthetic MCES",
    },

    "secondary_endpoints": [
        "paired sign test",
        "median MCES difference",
        "Morgan Tanimoto",
        "same-formula sensitivity",
    ],
}


META.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


print("=" * 90)
print("MSG-EG400 CONFIRMATORY FREEZE")
print("=" * 90)

print(
    "EG100 =",
    len(e100)
)

print(
    "EG500 =",
    len(e500)
)

print(
    "EG400 confirm =",
    len(confirm)
)

print(
    "Overlap EG100 =",
    len(
        set(confirm["connectivity_key"])
        &
        set(e100["connectivity_key"])
    )
)

print(
    "Union == EG500 =",
    (
        set(confirm["connectivity_key"])
        |
        set(e100["connectivity_key"])
    )
    ==
    set(e500["connectivity_key"])
)

print()
print(
    "CONFIRM HASH =",
    metadata["manifest_sha256"]
)

print()
print("DATA =", OUT_DATA)
print("META =", META)

print()
print("EG400 CONFIRMATORY FREEZE PASS")
