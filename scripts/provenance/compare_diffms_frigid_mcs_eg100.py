from pathlib import Path
from math import comb

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFMCS


D_PATH = Path(
    r"results\normalized\DIFFMS_K1_EG100.csv"
)

F_PATH = Path(
    r"results\normalized\FRIGID_BASE_FM1_EG100_HARMONIZED.csv"
)

OUT = Path(
    r"results\paired_DIFFMS_vs_FRIGID_EG100_MCS.csv"
)

SEED = 20261007
N_BOOT = 20000


def mol(s):
    if pd.isna(s):
        return None

    try:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None

        Chem.SanitizeMol(m)
        Chem.RemoveStereochemistry(m)

        return m

    except Exception:
        return None


def conn_key(m):
    try:
        k = Chem.MolToInchiKey(m)
        return k[:14] if k else None
    except Exception:
        return None


def run_mcs(a, b, bond_mode):

    res = rdFMCS.FindMCS(
        [a, b],

        atomCompare=
            rdFMCS.AtomCompare.CompareElements,

        bondCompare=bond_mode,

        ringMatchesRingOnly=True,

        completeRingsOnly=False,

        timeout=30,
    )

    return (
        int(res.numAtoms),
        int(res.numBonds),
        bool(res.canceled),
    )


def exact_sign_test(delta):

    x = np.asarray(delta, dtype=float)

    pos = int((x > 0).sum())
    neg = int((x < 0).sum())

    n = pos + neg

    if n == 0:
        return pos, neg, 1.0

    # Exact two-sided binomial sign test, p0=0.5.
    lo = min(pos, neg)

    tail = sum(
        comb(n, i)
        for i in range(0, lo + 1)
    ) / (2 ** n)

    p = min(
        1.0,
        2.0 * tail
    )

    return pos, neg, p


def bootstrap_ci(x):

    x = np.asarray(x, dtype=float)

    rng = np.random.default_rng(
        SEED
    )

    idx = rng.integers(
        0,
        len(x),
        size=(N_BOOT, len(x))
    )

    samples = x[idx]

    means = samples.mean(axis=1)
    medians = np.median(
        samples,
        axis=1
    )

    return {
        "mean":
            float(x.mean()),

        "mean_ci_low":
            float(
                np.quantile(
                    means,
                    0.025
                )
            ),

        "mean_ci_high":
            float(
                np.quantile(
                    means,
                    0.975
                )
            ),

        "median":
            float(
                np.median(x)
            ),

        "median_ci_low":
            float(
                np.quantile(
                    medians,
                    0.025
                )
            ),

        "median_ci_high":
            float(
                np.quantile(
                    medians,
                    0.975
                )
            ),
    }


d = pd.read_csv(D_PATH)
f = pd.read_csv(F_PATH)


d = d[
    [
        "panel_index",
        "spec",
        "truth_smiles",
        "pred_smiles",
        "task_valid",
        "same_formula",
        "exact_connectivity",
        "tanimoto",
    ]
].rename(
    columns={
        "truth_smiles":
            "d_truth_smiles",

        "pred_smiles":
            "d_pred_smiles",

        "task_valid":
            "d_valid",

        "same_formula":
            "d_formula",

        "exact_connectivity":
            "d_exact",

        "tanimoto":
            "d_tanimoto",
    }
)


f = f[
    [
        "panel_index",
        "spec",
        "truth_smiles",
        "pred_smiles",
        "task_valid",
        "same_formula",
        "exact_connectivity",
        "tanimoto",
    ]
].rename(
    columns={
        "truth_smiles":
            "f_truth_smiles",

        "pred_smiles":
            "f_pred_smiles",

        "task_valid":
            "f_valid",

        "same_formula":
            "f_formula",

        "exact_connectivity":
            "f_exact",

        "tanimoto":
            "f_tanimoto",
    }
)


x = d.merge(
    f,
    on=["panel_index", "spec"],
    validate="one_to_one",
)


x = x[
    x["d_valid"]
    &
    x["f_valid"]
    &
    ~x["d_exact"]
    &
    ~x["f_exact"]
].copy()


print("=" * 110)
print("DIFFMS vs FRIGID — PAIRED MCS ERROR GEOMETRY")
print("=" * 110)

print(
    "COMMON VALID WRONG =",
    len(x)
)


rows = []


for r in x.itertuples(index=False):

    truth_d = mol(
        r.d_truth_smiles
    )

    truth_f = mol(
        r.f_truth_smiles
    )

    pred_d = mol(
        r.d_pred_smiles
    )

    pred_f = mol(
        r.f_pred_smiles
    )


    if None in (
        truth_d,
        truth_f,
        pred_d,
        pred_f,
    ):
        raise RuntimeError(
            f"Parse failure: {r.spec}"
        )


    if conn_key(truth_d) != conn_key(truth_f):
        raise RuntimeError(
            f"Truth mismatch: {r.spec}"
        )


    truth_atoms = (
        truth_d.GetNumHeavyAtoms()
    )

    truth_bonds = (
        truth_d.GetNumBonds()
    )


    # ---------------------------------------------------
    # Bond-order-aware MCS
    # ---------------------------------------------------

    d_oa, d_ob, d_oc = run_mcs(
        truth_d,
        pred_d,
        rdFMCS.BondCompare.CompareOrder,
    )

    f_oa, f_ob, f_oc = run_mcs(
        truth_d,
        pred_f,
        rdFMCS.BondCompare.CompareOrder,
    )


    # ---------------------------------------------------
    # Pure topology MCS
    # ---------------------------------------------------

    d_ta, d_tb, d_tc = run_mcs(
        truth_d,
        pred_d,
        rdFMCS.BondCompare.CompareAny,
    )

    f_ta, f_tb, f_tc = run_mcs(
        truth_d,
        pred_f,
        rdFMCS.BondCompare.CompareAny,
    )


    rec = {
        "panel_index":
            r.panel_index,

        "spec":
            r.spec,

        "diffms_tanimoto":
            r.d_tanimoto,

        "frigid_tanimoto":
            r.f_tanimoto,

        "diffms_same_formula":
            r.d_formula,

        "frigid_same_formula":
            r.f_formula,

        "truth_heavy_atoms":
            truth_atoms,

        "truth_bonds":
            truth_bonds,


        "diffms_mcs_order_atom_fraction":
            d_oa / truth_atoms,

        "frigid_mcs_order_atom_fraction":
            f_oa / truth_atoms,

        "diffms_mcs_order_bond_fraction":
            d_ob / truth_bonds,

        "frigid_mcs_order_bond_fraction":
            f_ob / truth_bonds,


        "diffms_mcs_topology_atom_fraction":
            d_ta / truth_atoms,

        "frigid_mcs_topology_atom_fraction":
            f_ta / truth_atoms,

        "diffms_mcs_topology_bond_fraction":
            d_tb / truth_bonds,

        "frigid_mcs_topology_bond_fraction":
            f_tb / truth_bonds,


        "diffms_mcs_timeout":
            d_oc or d_tc,

        "frigid_mcs_timeout":
            f_oc or f_tc,
    }


    rec[
        "delta_order_atom_F_minus_D"
    ] = (
        rec[
            "frigid_mcs_order_atom_fraction"
        ]
        -
        rec[
            "diffms_mcs_order_atom_fraction"
        ]
    )


    rec[
        "delta_order_bond_F_minus_D"
    ] = (
        rec[
            "frigid_mcs_order_bond_fraction"
        ]
        -
        rec[
            "diffms_mcs_order_bond_fraction"
        ]
    )


    rec[
        "delta_topology_atom_F_minus_D"
    ] = (
        rec[
            "frigid_mcs_topology_atom_fraction"
        ]
        -
        rec[
            "diffms_mcs_topology_atom_fraction"
        ]
    )


    rec[
        "delta_topology_bond_F_minus_D"
    ] = (
        rec[
            "frigid_mcs_topology_bond_fraction"
        ]
        -
        rec[
            "diffms_mcs_topology_bond_fraction"
        ]
    )


    rows.append(rec)


out = pd.DataFrame(rows)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    OUT,
    index=False
)


print()
print(
    "MCS TIMEOUTS:",
    int(
        (
            out["diffms_mcs_timeout"]
            |
            out["frigid_mcs_timeout"]
        ).sum()
    )
)


metrics = [
    (
        "ORDER ATOM COVERAGE",
        "diffms_mcs_order_atom_fraction",
        "frigid_mcs_order_atom_fraction",
        "delta_order_atom_F_minus_D",
    ),
    (
        "ORDER BOND COVERAGE",
        "diffms_mcs_order_bond_fraction",
        "frigid_mcs_order_bond_fraction",
        "delta_order_bond_F_minus_D",
    ),
    (
        "TOPOLOGY ATOM COVERAGE",
        "diffms_mcs_topology_atom_fraction",
        "frigid_mcs_topology_atom_fraction",
        "delta_topology_atom_F_minus_D",
    ),
    (
        "TOPOLOGY BOND COVERAGE",
        "diffms_mcs_topology_bond_fraction",
        "frigid_mcs_topology_bond_fraction",
        "delta_topology_bond_F_minus_D",
    ),
]


for (
    label,
    dcol,
    fcol,
    delta_col,
) in metrics:

    print()
    print("=" * 110)
    print(label)
    print("=" * 110)

    print(
        "DiffMS mean/median =",
        out[dcol].mean(),
        "/",
        out[dcol].median()
    )

    print(
        "FRIGID mean/median =",
        out[fcol].mean(),
        "/",
        out[fcol].median()
    )


    delta = out[
        delta_col
    ].to_numpy()

    boot = bootstrap_ci(
        delta
    )

    pos, neg, p = exact_sign_test(
        delta
    )


    print(
        "Mean delta F-D =",
        boot["mean"]
    )

    print(
        "95% bootstrap CI mean =",
        (
            boot[
                "mean_ci_low"
            ],
            boot[
                "mean_ci_high"
            ]
        )
    )

    print(
        "Median delta F-D =",
        boot["median"]
    )

    print(
        "95% bootstrap CI median =",
        (
            boot[
                "median_ci_low"
            ],
            boot[
                "median_ci_high"
            ]
        )
    )

    print(
        "FRIGID better / DiffMS better =",
        pos,
        "/",
        neg
    )

    print(
        "Exact sign-test p =",
        p
    )


both_formula = out[
    out["diffms_same_formula"]
    &
    out["frigid_same_formula"]
]


print()
print("=" * 110)
print("SAME-FORMULA COMMON-WRONG SUBSET")
print("=" * 110)

print(
    "n =",
    len(both_formula)
)

print(
    both_formula[
        [
            "panel_index",
            "spec",
            "diffms_tanimoto",
            "frigid_tanimoto",
            "diffms_mcs_order_atom_fraction",
            "frigid_mcs_order_atom_fraction",
            "diffms_mcs_order_bond_fraction",
            "frigid_mcs_order_bond_fraction",
        ]
    ].to_string(
        index=False
    )
)


print()
print("Written:", OUT)
