from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import shutil

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcNumRings


# =============================================================================
# CONFIG
# =============================================================================

SOURCE = Path(r"C:\research\DiffMS\data\msg")
ROOT = Path(r"C:\research\ms_error_geometry")

MANIFEST_DIR = ROOT / "manifests"
DATA_DIR = ROOT / "data"

LABELS_FILE = SOURCE / "labels.tsv"
SPLIT_FILE = SOURCE / "split.tsv"

SPEC_DIR = SOURCE / "spec_files"

SUBFORMULA_DIR = (
    SOURCE
    / "subformulae"
    / "default_subformulae"
)

SEED = "MSG_ERROR_GEOMETRY_20261005_V01"

# Narrowest element vocabulary already verified for DiffMS.
# Hydrogen is generally implicit in RDKit molecular graphs.
ALLOWED_ELEMENTS = {
    "C",
    "N",
    "O",
    "P",
    "S",
    "F",
    "Cl",
}

PANEL_500 = 500
PANEL_100 = 100


# =============================================================================
# HELPERS
# =============================================================================

def stable_hash(*parts):
    text = "||".join(map(str, parts))

    return int.from_bytes(
        hashlib.sha256(
            text.encode("utf-8")
        ).digest()[:8],
        byteorder="big",
        signed=False,
    )


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def connectivity_key(inchikey):
    text = str(inchikey).strip()

    if len(text) < 14:
        return None

    return text[:14]


def inspect_structure(smiles):
    result = {
        "ok": False,
        "reason": None,
    }

    if pd.isna(smiles):
        result["reason"] = "missing_smiles"
        return result

    try:
        mol = Chem.MolFromSmiles(
            str(smiles)
        )
    except Exception:
        mol = None

    if mol is None:
        result["reason"] = "rdkit_parse_failed"
        return result

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        result["reason"] = "sanitize_failed"
        return result

    fragments = Chem.GetMolFrags(
        mol,
        asMols=False,
        sanitizeFrags=False,
    )

    if len(fragments) != 1:
        result["reason"] = "multi_fragment"
        return result

    if any(
        atom.GetFormalCharge() != 0
        for atom in mol.GetAtoms()
    ):
        result["reason"] = "formal_charge"
        return result

    if any(
        atom.GetIsotope() != 0
        for atom in mol.GetAtoms()
    ):
        result["reason"] = "isotope"
        return result

    elements = {
        atom.GetSymbol()
        for atom in mol.GetAtoms()
    }

    unsupported = (
        elements - ALLOWED_ELEMENTS
    )

    if unsupported:
        result["reason"] = (
            "unsupported_elements:"
            + ",".join(sorted(unsupported))
        )
        return result

    canonical = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=False,
    )

    result.update(
        {
            "ok": True,
            "reason": "PASS",
            "canonical_smiles": canonical,
            "heavy_atoms":
                mol.GetNumHeavyAtoms(),
            "rings":
                int(CalcNumRings(mol)),
            "mol_wt":
                float(
                    Descriptors.MolWt(mol)
                ),
            "elements":
                ",".join(
                    sorted(elements)
                ),
        }
    )

    return result


def heavy_atom_bin(n):
    if n <= 15:
        return "H00_15"
    if n <= 20:
        return "H16_20"
    if n <= 25:
        return "H21_25"
    if n <= 30:
        return "H26_30"
    if n <= 35:
        return "H31_35"
    if n <= 40:
        return "H36_40"
    if n <= 50:
        return "H41_50"

    return "H51_PLUS"


def ambiguity_bin(n):
    if n == 1:
        return "A1"
    if n <= 3:
        return "A2_3"
    if n <= 9:
        return "A4_9"

    return "A10_PLUS"


def allocate_quotas(counts, target_n):
    """
    Proportional allocation with largest-remainder correction.
    """

    total = int(counts.sum())

    if target_n > total:
        raise ValueError(
            f"Requested {target_n}, "
            f"but pool has only {total}"
        )

    raw = (
        counts.astype(float)
        / float(total)
        * target_n
    )

    quotas = raw.apply(
        math.floor
    ).astype(int)

    remaining = (
        target_n
        - int(quotas.sum())
    )

    fractional = (
        raw - quotas
    )

    order = sorted(
        counts.index,
        key=lambda s: (
            -fractional.loc[s],
            str(s),
        ),
    )

    while remaining > 0:

        changed = False

        for stratum in order:

            if remaining <= 0:
                break

            if (
                quotas.loc[stratum]
                < counts.loc[stratum]
            ):
                quotas.loc[stratum] += 1
                remaining -= 1
                changed = True

        if not changed:
            raise RuntimeError(
                "Could not allocate quotas"
            )

    assert int(quotas.sum()) == target_n

    return quotas


def stratified_hash_sample(
    df,
    n,
    panel_seed,
):
    counts = (
        df.groupby("stratum")
        .size()
        .sort_index()
    )

    quotas = allocate_quotas(
        counts,
        n,
    )

    chunks = []

    for stratum, quota in quotas.items():

        if quota == 0:
            continue

        part = df[
            df["stratum"] == stratum
        ].copy()

        part["_selection_hash"] = [
            stable_hash(
                SEED,
                panel_seed,
                r.connectivity_key,
                r.spec,
            )
            for r in part.itertuples(
                index=False
            )
        ]

        part = (
            part
            .sort_values(
                [
                    "_selection_hash",
                    "spec",
                ]
            )
            .head(quota)
        )

        chunks.append(part)

    out = pd.concat(
        chunks,
        ignore_index=True,
    )

    out["_final_hash"] = [
        stable_hash(
            SEED,
            panel_seed,
            r.connectivity_key,
            r.spec,
        )
        for r in out.itertuples(
            index=False
        )
    ]

    out = (
        out
        .sort_values(
            [
                "_final_hash",
                "spec",
            ]
        )
        .reset_index(drop=True)
    )

    out["panel_index"] = range(
        1,
        len(out) + 1
    )

    return out


def write_panel_dataset(
    panel,
    panel_name,
    labels,
):
    """
    Create a minimal MassSpecGym-compatible directory:
      labels.tsv
      split.tsv
      spec_files/
      subformulae/default_subformulae/
    """

    dst = DATA_DIR / panel_name

    if dst.exists():
        shutil.rmtree(dst)

    dst_spec = (
        dst / "spec_files"
    )

    dst_sub = (
        dst
        / "subformulae"
        / "default_subformulae"
    )

    dst_spec.mkdir(
        parents=True,
        exist_ok=True
    )

    dst_sub.mkdir(
        parents=True,
        exist_ok=True
    )

    selected_specs = set(
        panel["spec"]
    )

    labels_sub = labels[
        labels["spec"].isin(
            selected_specs
        )
    ].copy()

    order_map = {
        spec: i
        for i, spec
        in enumerate(panel["spec"])
    }

    labels_sub["_order"] = (
        labels_sub["spec"]
        .map(order_map)
    )

    labels_sub = (
        labels_sub
        .sort_values("_order")
        .drop(columns=["_order"])
    )

    labels_sub.to_csv(
        dst / "labels.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    split_sub = pd.DataFrame(
        {
            "name": panel["spec"],
            "split": ["test"]
            * len(panel),
        }
    )

    split_sub.to_csv(
        dst / "split.tsv",
        sep="\t",
        index=False,
        lineterminator="\n",
    )

    for spec in panel["spec"]:

        src_spec = (
            SPEC_DIR
            / f"{spec}.ms"
        )

        src_sub = (
            SUBFORMULA_DIR
            / f"{spec}.json"
        )

        shutil.copy2(
            src_spec,
            dst_spec / src_spec.name,
        )

        shutil.copy2(
            src_sub,
            dst_sub / src_sub.name,
        )

    return dst


def write_manifest(
    panel,
    filename,
):
    keep = [
        "panel_index",
        "spec",
        "connectivity_key",
        "inchikey",
        "formula",
        "smiles",
        "canonical_smiles",
        "heavy_atoms",
        "rings",
        "mol_wt",
        "elements",
        "formula_isomer_count",
        "heavy_atom_bin",
        "ambiguity_bin",
        "stratum",
    ]

    path = (
        MANIFEST_DIR / filename
    )

    panel[
        keep
    ].to_csv(
        path,
        index=False,
        lineterminator="\n",
    )

    return path


# =============================================================================
# INPUT
# =============================================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for required in [
    LABELS_FILE,
    SPLIT_FILE,
    SPEC_DIR,
    SUBFORMULA_DIR,
]:
    if not required.exists():
        raise FileNotFoundError(
            required
        )


labels = pd.read_csv(
    LABELS_FILE,
    sep="\t",
)

split = pd.read_csv(
    SPLIT_FILE,
    sep="\t",
)


required_label_cols = {
    "spec",
    "smiles",
    "inchikey",
    "formula",
}

required_split_cols = {
    "name",
    "split",
}


missing = (
    required_label_cols
    - set(labels.columns)
)

if missing:
    raise RuntimeError(
        "Missing labels columns: "
        + str(sorted(missing))
    )


missing = (
    required_split_cols
    - set(split.columns)
)

if missing:
    raise RuntimeError(
        "Missing split columns: "
        + str(sorted(missing))
    )


test_names = set(
    split.loc[
        split["split"]
        .astype(str)
        .str.lower()
        .eq("test"),
        "name",
    ]
)


raw = labels[
    labels["spec"].isin(
        test_names
    )
].copy()


print("=" * 110)
print("MASSPEC ERROR GEOMETRY PANEL BUILDER")
print("=" * 110)

print()
print("Test label rows =", len(raw))


# =============================================================================
# STRUCTURE + ARTIFACT AUDIT
# =============================================================================

records = []
rejections = []

for row in raw.itertuples(index=False):

    result = inspect_structure(
        row.smiles
    )

    ckey = connectivity_key(
        row.inchikey
    )

    if ckey is None:
        result["ok"] = False
        result["reason"] = (
            "invalid_inchikey"
        )

    spec_exists = (
        SPEC_DIR
        / f"{row.spec}.ms"
    ).exists()

    sub_exists = (
        SUBFORMULA_DIR
        / f"{row.spec}.json"
    ).exists()

    if (
        result["ok"]
        and not spec_exists
    ):
        result["ok"] = False
        result["reason"] = (
            "missing_spec_file"
        )

    if (
        result["ok"]
        and not sub_exists
    ):
        result["ok"] = False
        result["reason"] = (
            "missing_subformula"
        )

    if not result["ok"]:

        rejections.append(
            {
                "spec": row.spec,
                "inchikey": row.inchikey,
                "formula": row.formula,
                "reason":
                    result["reason"],
            }
        )

        continue

    records.append(
        {
            "spec": row.spec,
            "smiles": row.smiles,
            "inchikey": row.inchikey,
            "formula": row.formula,

            "connectivity_key":
                ckey,

            "canonical_smiles":
                result[
                    "canonical_smiles"
                ],

            "heavy_atoms":
                result[
                    "heavy_atoms"
                ],

            "rings":
                result[
                    "rings"
                ],

            "mol_wt":
                result[
                    "mol_wt"
                ],

            "elements":
                result[
                    "elements"
                ],
        }
    )


eligible_spectra = pd.DataFrame(
    records
)

rejections_df = pd.DataFrame(
    rejections
)


print(
    "Eligible spectrum rows =",
    len(eligible_spectra)
)

print(
    "Rejected spectrum rows =",
    len(rejections_df)
)


# =============================================================================
# ONE SPECTRUM PER CONNECTIVITY
# =============================================================================

eligible_spectra[
    "_rep_hash"
] = [
    stable_hash(
        SEED,
        "REPRESENTATIVE",
        r.connectivity_key,
        r.spec,
    )
    for r in eligible_spectra.itertuples(
        index=False
    )
]


eligible_unique = (
    eligible_spectra
    .sort_values(
        [
            "connectivity_key",
            "_rep_hash",
            "spec",
        ]
    )
    .drop_duplicates(
        "connectivity_key",
        keep="first",
    )
    .reset_index(drop=True)
)


# Formula ambiguity is measured among
# unique eligible molecular connectivities.

formula_counts = (
    eligible_unique
    .groupby("formula")
    ["connectivity_key"]
    .nunique()
)


eligible_unique[
    "formula_isomer_count"
] = (
    eligible_unique["formula"]
    .map(formula_counts)
    .astype(int)
)


eligible_unique[
    "heavy_atom_bin"
] = (
    eligible_unique["heavy_atoms"]
    .map(heavy_atom_bin)
)


eligible_unique[
    "ambiguity_bin"
] = (
    eligible_unique[
        "formula_isomer_count"
    ]
    .map(ambiguity_bin)
)


eligible_unique[
    "stratum"
] = (
    eligible_unique[
        "heavy_atom_bin"
    ]
    + "__"
    + eligible_unique[
        "ambiguity_bin"
    ]
)


if len(eligible_unique) < PANEL_500:
    raise RuntimeError(
        "Eligible unique pool "
        f"has only {len(eligible_unique)} "
        "molecules."
    )


print(
    "Eligible unique connectivity pool =",
    len(eligible_unique)
)


# =============================================================================
# BUILD EG500, THEN NESTED EG100
# =============================================================================

eg500 = stratified_hash_sample(
    eligible_unique,
    PANEL_500,
    "EG500_PREFREEZE",
)


eg100 = stratified_hash_sample(
    eg500,
    PANEL_100,
    "EG100_DEV",
)


assert len(eg500) == 500
assert len(eg100) == 100

assert (
    eg500[
        "connectivity_key"
    ].nunique()
    == 500
)

assert (
    eg100[
        "connectivity_key"
    ].nunique()
    == 100
)

assert set(
    eg100["connectivity_key"]
).issubset(
    set(eg500["connectivity_key"])
)


# =============================================================================
# WRITE MANIFESTS
# =============================================================================

pool_path = (
    MANIFEST_DIR
    / "msg_eligible_unique_pool_v01.csv"
)

eligible_unique[
    [
        "spec",
        "connectivity_key",
        "inchikey",
        "formula",
        "smiles",
        "canonical_smiles",
        "heavy_atoms",
        "rings",
        "mol_wt",
        "elements",
        "formula_isomer_count",
        "heavy_atom_bin",
        "ambiguity_bin",
        "stratum",
    ]
].to_csv(
    pool_path,
    index=False,
    lineterminator="\n",
)


reject_path = (
    MANIFEST_DIR
    / "msg_panel_rejections_v01.csv"
)

rejections_df.to_csv(
    reject_path,
    index=False,
    lineterminator="\n",
)


eg500_path = write_manifest(
    eg500,
    "MSG_EG500_PREFREEZE_v01.csv",
)

eg100_path = write_manifest(
    eg100,
    "MSG_EG100_DEV_v01.csv",
)


# =============================================================================
# CREATE DATA DIRECTORIES
# =============================================================================

eg500_data = write_panel_dataset(
    eg500,
    "MSG_EG500_PREFREEZE_v01",
    labels,
)

eg100_data = write_panel_dataset(
    eg100,
    "MSG_EG100_DEV_v01",
    labels,
)


# =============================================================================
# FREEZE METADATA
# =============================================================================

meta = {
    "panel_version":
        "v0.1",

    "status":
        "PREFREEZE",

    "seed":
        SEED,

    "source_dataset":
        "MassSpecGym v1.5 local MSG preprocessing",

    "source_path":
        str(SOURCE),

    "selection": {
        "split":
            "test",

        "unique_unit":
            "first 14 InChIKey characters",

        "single_spectrum_per_connectivity":
            True,

        "allowed_elements":
            sorted(ALLOWED_ELEMENTS),

        "formal_charge":
            "zero only",

        "isotopes":
            "excluded",

        "multi_fragment":
            "excluded",

        "artifacts_required": [
            "spec_files/<spec>.ms",
            (
                "subformulae/"
                "default_subformulae/"
                "<spec>.json"
            ),
        ],

        "stratification": [
            "heavy_atom_bin",
            "formula_ambiguity_bin",
        ],
    },

    "counts": {
        "test_label_rows":
            len(raw),

        "eligible_spectrum_rows":
            len(eligible_spectra),

        "eligible_unique_connectivities":
            len(eligible_unique),

        "EG500":
            len(eg500),

        "EG100":
            len(eg100),
    },

    "hashes": {
        "eligible_pool_sha256":
            sha256_file(pool_path),

        "EG500_manifest_sha256":
            sha256_file(eg500_path),

        "EG100_manifest_sha256":
            sha256_file(eg100_path),
    },
}


meta_path = (
    MANIFEST_DIR
    / "MSG_EG_PANEL_FREEZE_METADATA_v01.json"
)

meta_path.write_text(
    json.dumps(
        meta,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)


# =============================================================================
# REPORT
# =============================================================================

def panel_summary(
    name,
    df,
):
    print()
    print("=" * 110)
    print(name)
    print("=" * 110)

    print(
        "n =",
        len(df)
    )

    print(
        "unique connectivity =",
        df[
            "connectivity_key"
        ].nunique()
    )

    print(
        "heavy atoms "
        "mean/median/min/max =",
        round(
            df["heavy_atoms"].mean(),
            2
        ),
        "/",
        round(
            df["heavy_atoms"].median(),
            2
        ),
        "/",
        int(
            df["heavy_atoms"].min()
        ),
        "/",
        int(
            df["heavy_atoms"].max()
        ),
    )

    print(
        "rings "
        "mean/median/max =",
        round(
            df["rings"].mean(),
            2
        ),
        "/",
        round(
            df["rings"].median(),
            2
        ),
        "/",
        int(
            df["rings"].max()
        ),
    )

    print(
        "formula isomer count "
        "median/max =",
        round(
            df[
                "formula_isomer_count"
            ].median(),
            2
        ),
        "/",
        int(
            df[
                "formula_isomer_count"
            ].max()
        ),
    )

    print()
    print(
        df[
            "heavy_atom_bin"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        df[
            "ambiguity_bin"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


panel_summary(
    "ELIGIBLE UNIQUE POOL",
    eligible_unique,
)

panel_summary(
    "MSG-EG500 PREFREEZE",
    eg500,
)

panel_summary(
    "MSG-EG100 DEV",
    eg100,
)


print()
print("=" * 110)
print("FREEZE HASHES")
print("=" * 110)

print(
    "EG500 =",
    meta["hashes"][
        "EG500_manifest_sha256"
    ]
)

print(
    "EG100 =",
    meta["hashes"][
        "EG100_manifest_sha256"
    ]
)

print()
print(
    "EG100 subset EG500 =",
    set(
        eg100[
            "connectivity_key"
        ]
    ).issubset(
        set(
            eg500[
                "connectivity_key"
            ]
        )
    )
)

print()
print("EG100 data =", eg100_data)
print("EG500 data =", eg500_data)

print()
print("Metadata =", meta_path)

print()
print("PANEL BUILD PASS")
