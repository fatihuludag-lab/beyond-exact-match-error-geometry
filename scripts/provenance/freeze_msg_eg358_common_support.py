from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd


ROOT = Path(r"C:\research\ms_error_geometry")

MASTER_MANIFEST = (
    ROOT / "manifests" /
    "MSG_EG400_CONFIRM_v01.csv"
)

MASTER_DATA = (
    ROOT / "data" /
    "MSG_EG400_CONFIRM_v01"
)

OUT_MANIFEST = (
    ROOT / "manifests" /
    "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv"
)

OUT_EXCLUDED = (
    ROOT / "manifests" /
    "MSG_EG400_EXCLUDED_UNSUPPORTED_ADDUCTS_v01.csv"
)

OUT_DATA = (
    ROOT / "data" /
    "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01"
)

OUT_META = (
    ROOT / "manifests" /
    "MSG_EG358_COMMON_SUPPORT_FREEZE_v01.json"
)


ALLOWED = {
    "[M+H]+",
    "[M-H]-",
}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


manifest = pd.read_csv(
    MASTER_MANIFEST
)

labels = pd.read_csv(
    MASTER_DATA / "labels.tsv",
    sep="\t"
)


assert len(manifest) == 400
assert len(labels) == 400
assert manifest["spec"].nunique() == 400
assert labels["spec"].nunique() == 400


# ------------------------------------------------------------
# Determine support using INPUT METADATA ONLY.
# No model outputs are used here.
# ------------------------------------------------------------

support_table = labels[
    ["spec", "ionization"]
].copy()

support_table["supported"] = (
    support_table["ionization"]
    .isin(ALLOWED)
)


supported_specs = set(
    support_table.loc[
        support_table["supported"],
        "spec"
    ]
)

unsupported_specs = set(
    support_table.loc[
        ~support_table["supported"],
        "spec"
    ]
)


confirm = manifest[
    manifest["spec"].isin(
        supported_specs
    )
].copy()

excluded = (
    manifest[
        manifest["spec"].isin(
            unsupported_specs
        )
    ]
    .merge(
        support_table[
            ["spec", "ionization"]
        ],
        on="spec",
        how="left",
        validate="one_to_one",
    )
)


# Preserve original EG400 ordering.
confirm["_master_order"] = (
    confirm["spec"]
    .map(
        {
            spec: i
            for i, spec
            in enumerate(
                manifest["spec"]
            )
        }
    )
)

confirm = (
    confirm
    .sort_values("_master_order")
    .drop(columns="_master_order")
    .reset_index(drop=True)
)

confirm[
    "common_support_index"
] = range(
    1,
    len(confirm) + 1
)


# ------------------------------------------------------------
# Hard gates
# ------------------------------------------------------------

assert len(confirm) == 358, len(confirm)
assert len(excluded) == 42, len(excluded)

assert set(confirm["spec"]).isdisjoint(
    set(excluded["spec"])
)

assert (
    set(confirm["spec"])
    |
    set(excluded["spec"])
) == set(manifest["spec"])

assert (
    confirm["spec"].nunique()
    == 358
)


confirm.to_csv(
    OUT_MANIFEST,
    index=False,
    lineterminator="\n"
)

excluded.to_csv(
    OUT_EXCLUDED,
    index=False,
    lineterminator="\n"
)


# ------------------------------------------------------------
# Build minimal MassSpecGym directory
# ------------------------------------------------------------

if OUT_DATA.exists():
    shutil.rmtree(
        OUT_DATA
    )

spec_out = (
    OUT_DATA /
    "spec_files"
)

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


order = {
    spec: i
    for i, spec
    in enumerate(
        confirm["spec"]
    )
}

labels_out = labels[
    labels["spec"].isin(
        supported_specs
    )
].copy()

labels_out["_order"] = (
    labels_out["spec"]
    .map(order)
)

labels_out = (
    labels_out
    .sort_values("_order")
    .drop(columns="_order")
)


assert set(
    labels_out["ionization"]
).issubset(ALLOWED)

assert len(labels_out) == 358


labels_out.to_csv(
    OUT_DATA / "labels.tsv",
    sep="\t",
    index=False,
    lineterminator="\n"
)


pd.DataFrame({
    "name":
        confirm["spec"],

    "split":
        ["test"] * len(confirm),
}).to_csv(
    OUT_DATA / "split.tsv",
    sep="\t",
    index=False,
    lineterminator="\n"
)


for spec in confirm["spec"]:

    src_spec = (
        MASTER_DATA /
        "spec_files" /
        f"{spec}.ms"
    )

    src_sub = (
        MASTER_DATA /
        "subformulae" /
        "default_subformulae" /
        f"{spec}.json"
    )

    if not src_spec.exists():
        raise FileNotFoundError(
            src_spec
        )

    if not src_sub.exists():
        raise FileNotFoundError(
            src_sub
        )

    shutil.copy2(
        src_spec,
        spec_out /
        src_spec.name,
    )

    shutil.copy2(
        src_sub,
        sub_out /
        src_sub.name,
    )


new_hash = sha256(
    OUT_MANIFEST
)


metadata = {
    "panel":
        "MSG-EG358-COMMON-SUPPORT-CONFIRM-v01",

    "status":
        "FROZEN_CONFIRMATORY_COMMON_SUPPORT",

    "master_confirmatory_panel":
        "MSG-EG400-CONFIRM-v01",

    "master_manifest_sha256":
        "c644b79ba80d36661338bf8224e185b7b47cc2c34652478e7eb03c85e4405bde",

    "n_master":
        400,

    "n_confirmatory":
        358,

    "n_excluded":
        42,

    "supported_ionization":
        sorted(ALLOWED),

    "observed_supported_counts":
        labels_out[
            "ionization"
        ].value_counts().to_dict(),

    "excluded_counts":
        excluded[
            "ionization"
        ].value_counts().to_dict(),

    "exclusion_rule":
        "Ionization/adduct unsupported by at least one model in the three-model confirmatory comparison.",

    "selection_basis":
        "Input metadata only; no confirmatory model outputs inspected.",

    "timing":
        "Compatibility refinement performed before confirmatory model inference.",

    "manifest_sha256":
        new_hash,
}


OUT_META.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


print("=" * 100)
print("MSG-EG358 COMMON-SUPPORT CONFIRMATORY FREEZE")
print("=" * 100)

print(
    "MASTER =",
    len(manifest)
)

print(
    "SUPPORTED =",
    len(confirm)
)

print(
    "EXCLUDED =",
    len(excluded)
)

print()
print("SUPPORTED IONIZATION")

print(
    labels_out[
        "ionization"
    ].value_counts().to_string()
)

print()
print("EXCLUDED IONIZATION")

print(
    excluded[
        "ionization"
    ].value_counts().to_string()
)

print()
print(
    "INTERSECTION =",
    len(
        set(confirm["spec"])
        &
        set(excluded["spec"])
    )
)

print(
    "UNION == MASTER =",
    (
        set(confirm["spec"])
        |
        set(excluded["spec"])
    )
    ==
    set(manifest["spec"])
)

print()
print(
    "CONFIRM HASH =",
    new_hash
)

print()
print(
    "DATA =",
    OUT_DATA
)

print(
    "META =",
    OUT_META
)

print()
print(
    "EG358 COMMON-SUPPORT FREEZE PASS"
)
