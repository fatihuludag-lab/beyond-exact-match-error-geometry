from pathlib import Path
import pickle
from collections import Counter, defaultdict

import pandas as pd
from rdkit import Chem


MANIFEST = Path(
    r"C:\research\ms_error_geometry\manifests\MSG_EG100_DEV_v01.csv"
)

TRUE_PKL = Path(
    r"C:\research\ms_error_geometry\results"
    r"\diffms_eg100_k1\preds"
    r"\MSG_EG100_DIFFMS_K1_rank_0_true_0.pkl"
)


def prep_smiles(s):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        return None

    Chem.SanitizeMol(m)
    return m


def fresh(m):
    return Chem.Mol(m.ToBinary())


def full_key(m):
    try:
        x = fresh(m)
        return Chem.MolToInchiKey(x)
    except Exception:
        return None


def conn_key(m):
    k = full_key(m)
    return k[:14] if k else None


def can_iso(m):
    try:
        return Chem.MolToSmiles(
            fresh(m),
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


def can_no_stereo(m):
    try:
        x = fresh(m)
        Chem.RemoveStereochemistry(x)

        return Chem.MolToSmiles(
            x,
            canonical=True,
            isomericSmiles=False,
        )
    except Exception:
        return None


manifest = pd.read_csv(MANIFEST)

with TRUE_PKL.open("rb") as f:
    truths = pickle.load(f)


# ----------------------------------------------------------------------
# Manifest-side RDKit identity audit
# ----------------------------------------------------------------------

mrows = []

for r in manifest.itertuples(index=False):

    m = prep_smiles(r.smiles)

    mrows.append({
        "panel_index": r.panel_index,
        "spec": r.spec,

        "label_full_inchikey":
            str(r.inchikey),

        "label_connectivity_key":
            str(r.connectivity_key),

        "rdkit_full_inchikey":
            full_key(m),

        "rdkit_connectivity_key":
            conn_key(m),

        "canonical_iso":
            can_iso(m),

        "canonical_nostereo":
            can_no_stereo(m),
    })


mdf = pd.DataFrame(mrows)

print("=" * 110)
print("MANIFEST IDENTITY AUDIT")
print("=" * 110)

print(
    "n =",
    len(mdf)
)

print(
    "label connectivity unique =",
    mdf["label_connectivity_key"].nunique()
)

print(
    "RDKit connectivity unique =",
    mdf["rdkit_connectivity_key"].nunique()
)

print(
    "label key == RDKit key =",
    int(
        (
            mdf["label_connectivity_key"]
            ==
            mdf["rdkit_connectivity_key"]
        ).sum()
    ),
    "/",
    len(mdf)
)


dup_manifest = mdf[
    mdf["rdkit_connectivity_key"]
    .duplicated(keep=False)
].sort_values(
    "rdkit_connectivity_key"
)

print()
print("MANIFEST RDKit-CONNECTIVITY DUPLICATES")
print(
    dup_manifest[
        [
            "panel_index",
            "spec",
            "label_connectivity_key",
            "rdkit_connectivity_key",
            "canonical_nostereo",
        ]
    ].to_string(index=False)
    if len(dup_manifest)
    else "NONE"
)


# ----------------------------------------------------------------------
# Pickle truth identity audit
# ----------------------------------------------------------------------

trows = []

for i, mol in enumerate(truths):

    trows.append({
        "pickle_index": i,

        "full_inchikey":
            full_key(mol),

        "connectivity_key":
            conn_key(mol),

        "canonical_iso":
            can_iso(mol),

        "canonical_nostereo":
            can_no_stereo(mol),
    })


tdf = pd.DataFrame(trows)

print()
print("=" * 110)
print("PERSISTED TRUTH AUDIT")
print("=" * 110)

print(
    "n =",
    len(tdf)
)

print(
    "full InChIKey unique =",
    tdf["full_inchikey"].nunique()
)

print(
    "connectivity key unique =",
    tdf["connectivity_key"].nunique()
)

print(
    "canonical isomeric SMILES unique =",
    tdf["canonical_iso"].nunique()
)

print(
    "canonical no-stereo SMILES unique =",
    tdf["canonical_nostereo"].nunique()
)


dup_truth = tdf[
    tdf["connectivity_key"]
    .duplicated(keep=False)
].sort_values(
    "connectivity_key"
)

print()
print("PICKLE TRUTH CONNECTIVITY DUPLICATES")

print(
    dup_truth.to_string(index=False)
    if len(dup_truth)
    else "NONE"
)


# ----------------------------------------------------------------------
# Test candidate identity schemes
# ----------------------------------------------------------------------

for col_t, col_m, name in [
    (
        "connectivity_key",
        "rdkit_connectivity_key",
        "RDKit connectivity key",
    ),
    (
        "full_inchikey",
        "rdkit_full_inchikey",
        "full RDKit InChIKey",
    ),
    (
        "canonical_iso",
        "canonical_iso",
        "canonical isomeric SMILES",
    ),
    (
        "canonical_nostereo",
        "canonical_nostereo",
        "canonical no-stereo SMILES",
    ),
]:

    manifest_counts = Counter(
        mdf[col_m]
    )

    uniquely_mappable = sum(
        manifest_counts[x] == 1
        for x in tdf[col_t]
    )

    present = sum(
        x in manifest_counts
        for x in tdf[col_t]
    )

    print()
    print(name)
    print(
        "present in manifest =",
        present,
        "/100"
    )
    print(
        "uniquely mappable =",
        uniquely_mappable,
        "/100"
    )


out1 = Path(
    r"C:\research\ms_error_geometry\results"
    r"\diffms_eg100_manifest_identity_audit.csv"
)

out2 = Path(
    r"C:\research\ms_error_geometry\results"
    r"\diffms_eg100_pickle_truth_identity_audit.csv"
)

mdf.to_csv(out1, index=False)
tdf.to_csv(out2, index=False)

print()
print("Manifest audit =", out1)
print("Truth audit    =", out2)
