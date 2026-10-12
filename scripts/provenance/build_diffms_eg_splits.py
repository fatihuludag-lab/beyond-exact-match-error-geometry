from pathlib import Path
import pandas as pd

ORIG = Path(
    r"C:\research\DiffMS\data\msg\split.tsv"
)

MANIFEST = Path(
    r"C:\research\ms_error_geometry\manifests\MSG_EG100_DEV_v01.csv"
)

OUT10 = Path(
    r"C:\research\ms_error_geometry\manifests\DIFFMS_MSG_EG10_split.tsv"
)

OUT100 = Path(
    r"C:\research\ms_error_geometry\manifests\DIFFMS_MSG_EG100_split.tsv"
)

split = pd.read_csv(
    ORIG,
    sep="\t"
)

manifest = pd.read_csv(
    MANIFEST
)

eg100 = set(
    manifest["spec"]
)

eg10 = set(
    manifest
    .sort_values("panel_index")
    .head(10)["spec"]
)


def build(selected, output):

    x = split.copy()

    original_test = (
        x["split"]
        .astype(str)
        .str.lower()
        .eq("test")
    )

    # Keep original train + val unchanged.
    #
    # Non-selected original-test entries become val ONLY
    # so they disappear from evaluation; fixed pretrained
    # weights mean they are never used for training here.
    x.loc[
        original_test,
        "split"
    ] = "val"

    x.loc[
        x["name"].isin(selected),
        "split"
    ] = "test"

    found = set(
        x.loc[
            x["split"] == "test",
            "name"
        ]
    )

    missing = selected - found

    if missing:
        raise RuntimeError(
            f"Missing specs: {sorted(missing)}"
        )

    assert len(found) == len(selected)

    x.to_csv(
        output,
        sep="\t",
        index=False,
        lineterminator="\n"
    )

    print()
    print(output)
    print(
        x["split"]
        .value_counts()
        .to_string()
    )

    print(
        "TEST =",
        len(found)
    )


build(
    eg10,
    OUT10
)

build(
    eg100,
    OUT100
)

print()
print("PASS")
