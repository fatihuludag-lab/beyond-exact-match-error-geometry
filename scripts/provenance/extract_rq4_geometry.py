from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS, rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import CalcMolFormula, CalcNumRings


REAL = Path(
    r"results\RQ4_REAL_SAME_FORMULA_WRONG_EG100.csv"
)

SYN = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_PRIMARY.parquet"
)

OUT_REAL = Path(
    r"results\RQ4_REAL_GEOMETRY_EG100.csv"
)

OUT_SYN = Path(
    r"results\RQ4_SYNTHETIC_GEOMETRY_EG100.csv"
)


def prep(smiles):

    try:
        m = Chem.MolFromSmiles(
            str(smiles)
        )

        if m is None:
            return None

        Chem.SanitizeMol(m)
        Chem.RemoveStereochemistry(m)

        return m

    except Exception:
        return None


def fingerprint(m):

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )

    return gen.GetFingerprint(m)


def compute_geometry(task):

    (
        source,
        parent_spec,
        observation_id,
        model_or_family,
        truth_smiles,
        candidate_smiles,
        timeout,
    ) = task

    truth = prep(truth_smiles)
    cand = prep(candidate_smiles)

    if truth is None or cand is None:

        return {
            "source": source,
            "parent_spec": parent_spec,
            "observation_id": observation_id,
            "model_or_family": model_or_family,
            "valid": False,
        }


    truth_formula = CalcMolFormula(truth)
    cand_formula = CalcMolFormula(cand)

    same_formula = (
        truth_formula == cand_formula
    )


    tani = float(
        DataStructs.TanimotoSimilarity(
            fingerprint(truth),
            fingerprint(cand),
        )
    )


    order = rdFMCS.FindMCS(
        [truth, cand],
        atomCompare=
            rdFMCS.AtomCompare.CompareElements,
        bondCompare=
            rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        timeout=timeout,
    )


    topo = rdFMCS.FindMCS(
        [truth, cand],
        atomCompare=
            rdFMCS.AtomCompare.CompareElements,
        bondCompare=
            rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
        timeout=timeout,
    )


    n_atoms = truth.GetNumHeavyAtoms()
    n_bonds = truth.GetNumBonds()

    return {
        "source":
            source,

        "parent_spec":
            parent_spec,

        "observation_id":
            observation_id,

        "model_or_family":
            model_or_family,

        "valid":
            True,

        "same_formula":
            same_formula,

        "truth_formula":
            truth_formula,

        "candidate_formula":
            cand_formula,

        "truth_heavy_atoms":
            n_atoms,

        "truth_bonds":
            n_bonds,

        "truth_rings":
            int(CalcNumRings(truth)),

        "candidate_rings":
            int(CalcNumRings(cand)),

        "abs_ring_diff":
            abs(
                int(CalcNumRings(truth))
                -
                int(CalcNumRings(cand))
            ),

        "tanimoto":
            tani,

        "mcs_order_atoms":
            int(order.numAtoms),

        "mcs_order_bonds":
            int(order.numBonds),

        "mcs_order_atom_fraction":
            (
                order.numAtoms / n_atoms
                if n_atoms else 0.0
            ),

        "mcs_order_bond_fraction":
            (
                order.numBonds / n_bonds
                if n_bonds else 0.0
            ),

        "mcs_topology_atoms":
            int(topo.numAtoms),

        "mcs_topology_bonds":
            int(topo.numBonds),

        "mcs_topology_atom_fraction":
            (
                topo.numAtoms / n_atoms
                if n_atoms else 0.0
            ),

        "mcs_topology_bond_fraction":
            (
                topo.numBonds / n_bonds
                if n_bonds else 0.0
            ),

        "mcs_order_timeout":
            bool(order.canceled),

        "mcs_topology_timeout":
            bool(topo.canceled),
    }


def run_tasks(tasks, workers):

    rows = []

    total = len(tasks)

    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = [
            executor.submit(
                compute_geometry,
                task
            )
            for task in tasks
        ]

        for i, future in enumerate(
            as_completed(futures),
            start=1
        ):

            rows.append(
                future.result()
            )

            if (
                i % 100 == 0
                or i == total
            ):
                print(
                    f"{i}/{total} completed"
                )

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
    )

    args = parser.parse_args()


    real = pd.read_csv(REAL)

    syn = pd.read_parquet(SYN)


    print("=" * 100)
    print("RQ4 ERROR-GEOMETRY EXTRACTION")
    print("=" * 100)

    print(
        "REAL =",
        len(real)
    )

    print(
        "SYNTHETIC =",
        len(syn)
    )


    real_tasks = []

    for i, r in enumerate(
        real.itertuples(index=False)
    ):

        real_tasks.append(
            (
                "real",
                str(r.spec),
                f"REAL_{i:04d}",
                str(r.model),
                str(r.truth_smiles),
                str(r.pred_smiles),
                args.timeout,
            )
        )


    syn_tasks = []

    for r in syn.itertuples(index=False):

        syn_tasks.append(
            (
                "synthetic",
                str(r.parent_spec),
                str(r.sample_id),
                str(r.corruption_family),
                str(r.true_smiles),
                str(r.start_smiles),
                args.timeout,
            )
        )


    print()
    print("Processing real errors...")

    real_out = run_tasks(
        real_tasks,
        args.workers,
    )


    print()
    print("Processing synthetic errors...")

    syn_out = run_tasks(
        syn_tasks,
        args.workers,
    )


    OUT_REAL.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    real_out.to_csv(
        OUT_REAL,
        index=False
    )

    syn_out.to_csv(
        OUT_SYN,
        index=False
    )


    print()
    print("=" * 100)
    print("AUDIT")
    print("=" * 100)


    for name, df in [
        ("REAL", real_out),
        ("SYNTHETIC", syn_out),
    ]:

        print()
        print(name)

        print(
            "n =",
            len(df)
        )

        print(
            "valid =",
            int(df["valid"].sum())
        )

        print(
            "same formula =",
            int(df["same_formula"].sum())
        )

        print(
            "MCS order timeouts =",
            int(
                df[
                    "mcs_order_timeout"
                ].sum()
            )
        )

        print(
            "MCS topology timeouts =",
            int(
                df[
                    "mcs_topology_timeout"
                ].sum()
            )
        )

        print(
            "Tanimoto mean/median =",
            df["tanimoto"].mean(),
            "/",
            df["tanimoto"].median()
        )

        print(
            "Order atom coverage mean/median =",
            df[
                "mcs_order_atom_fraction"
            ].mean(),
            "/",
            df[
                "mcs_order_atom_fraction"
            ].median()
        )


    print()
    print("REAL =", OUT_REAL)
    print("SYN  =", OUT_SYN)

    print()
    print("RQ4 GEOMETRY EXTRACTION PASS")


if __name__ == "__main__":
    main()
