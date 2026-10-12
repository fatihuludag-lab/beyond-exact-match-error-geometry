from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdFMCS


REAL_SRC = Path(
    r"results\RQ4_REAL_SAME_FORMULA_WRONG_EG100.csv"
)

SYN_SRC = Path(
    r"results\RQ4_SYNTHETIC_MATCHED_EG100_PRIMARY.parquet"
)

REAL_GEOM = Path(
    r"results\RQ4_REAL_GEOMETRY_EG100.csv"
)

SYN_GEOM = Path(
    r"results\RQ4_SYNTHETIC_GEOMETRY_EG100.csv"
)

OUT_REAL = Path(
    r"results\RQ4_REAL_GEOMETRY_EG100_CLEAN.csv"
)

OUT_SYN = Path(
    r"results\RQ4_SYNTHETIC_GEOMETRY_EG100_CLEAN.csv"
)


def prep(smiles):
    m = Chem.MolFromSmiles(str(smiles))

    if m is None:
        raise ValueError(smiles)

    Chem.SanitizeMol(m)
    Chem.RemoveStereochemistry(m)

    return m


def compute_one(task):

    (
        observation_id,
        truth_smiles,
        candidate_smiles,
        need_order,
        need_topology,
        timeout,
    ) = task

    truth = prep(truth_smiles)
    cand = prep(candidate_smiles)

    na = truth.GetNumHeavyAtoms()
    nb = truth.GetNumBonds()

    result = {
        "observation_id":
            observation_id
    }

    if need_order:

        r = rdFMCS.FindMCS(
            [truth, cand],
            atomCompare=
                rdFMCS.AtomCompare.CompareElements,
            bondCompare=
                rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            timeout=timeout,
        )

        result.update({
            "mcs_order_atoms":
                int(r.numAtoms),

            "mcs_order_bonds":
                int(r.numBonds),

            "mcs_order_atom_fraction":
                r.numAtoms / na,

            "mcs_order_bond_fraction":
                r.numBonds / nb
                if nb else 0.0,

            "mcs_order_timeout":
                bool(r.canceled),
        })

    if need_topology:

        r = rdFMCS.FindMCS(
            [truth, cand],
            atomCompare=
                rdFMCS.AtomCompare.CompareElements,
            bondCompare=
                rdFMCS.BondCompare.CompareAny,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            timeout=timeout,
        )

        result.update({
            "mcs_topology_atoms":
                int(r.numAtoms),

            "mcs_topology_bonds":
                int(r.numBonds),

            "mcs_topology_atom_fraction":
                r.numAtoms / na,

            "mcs_topology_bond_fraction":
                r.numBonds / nb
                if nb else 0.0,

            "mcs_topology_timeout":
                bool(r.canceled),
        })

    return result


def retry_dataset(
    geom,
    lookup,
    output,
    workers,
    timeout,
):

    bad = geom[
        geom["mcs_order_timeout"]
        |
        geom["mcs_topology_timeout"]
    ].copy()

    print()
    print(
        "Timeout rows before retry =",
        len(bad)
    )

    tasks = []

    for r in bad.itertuples(index=False):

        src = lookup.loc[
            r.observation_id
        ]

        tasks.append(
            (
                r.observation_id,
                src["truth_smiles"],
                src["candidate_smiles"],
                bool(r.mcs_order_timeout),
                bool(r.mcs_topology_timeout),
                timeout,
            )
        )

    results = []

    with ProcessPoolExecutor(
        max_workers=workers
    ) as ex:

        futures = [
            ex.submit(
                compute_one,
                t
            )
            for t in tasks
        ]

        for i, f in enumerate(
            as_completed(futures),
            start=1
        ):
            results.append(
                f.result()
            )

            print(
                f"{i}/{len(tasks)} retries completed"
            )

    geom = geom.copy()
    geom = geom.set_index(
        "observation_id",
        drop=False
    )

    for res in results:

        oid = res[
            "observation_id"
        ]

        for col, value in res.items():

            if col == "observation_id":
                continue

            geom.loc[
                oid,
                col
            ] = value

    geom = geom.reset_index(
        drop=True
    )

    print(
        "Order timeouts after retry =",
        int(
            geom[
                "mcs_order_timeout"
            ].sum()
        )
    )

    print(
        "Topology timeouts after retry =",
        int(
            geom[
                "mcs_topology_timeout"
            ].sum()
        )
    )

    geom.to_csv(
        output,
        index=False
    )

    print(
        "Written:",
        output
    )

    return geom


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--timeout",
        type=int,
        default=600
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4
    )

    args = parser.parse_args()

    real_src = pd.read_csv(
        REAL_SRC
    )

    real_src[
        "observation_id"
    ] = [
        f"REAL_{i:04d}"
        for i in range(
            len(real_src)
        )
    ]

    real_lookup = (
        real_src[
            [
                "observation_id",
                "truth_smiles",
                "pred_smiles",
            ]
        ]
        .rename(
            columns={
                "pred_smiles":
                    "candidate_smiles"
            }
        )
        .set_index(
            "observation_id"
        )
    )


    syn_src = pd.read_parquet(
        SYN_SRC
    )

    syn_lookup = (
        syn_src[
            [
                "sample_id",
                "true_smiles",
                "start_smiles",
            ]
        ]
        .rename(
            columns={
                "sample_id":
                    "observation_id",

                "true_smiles":
                    "truth_smiles",

                "start_smiles":
                    "candidate_smiles",
            }
        )
        .set_index(
            "observation_id"
        )
    )


    real_geom = pd.read_csv(
        REAL_GEOM
    )

    syn_geom = pd.read_csv(
        SYN_GEOM
    )


    print("=" * 100)
    print("REAL TIMEOUT RETRY")
    print("=" * 100)

    real_clean = retry_dataset(
        real_geom,
        real_lookup,
        OUT_REAL,
        args.workers,
        args.timeout,
    )


    print()
    print("=" * 100)
    print("SYNTHETIC TIMEOUT RETRY")
    print("=" * 100)

    syn_clean = retry_dataset(
        syn_geom,
        syn_lookup,
        OUT_SYN,
        args.workers,
        args.timeout,
    )


    total_left = (
        int(
            real_clean[
                "mcs_order_timeout"
            ].sum()
        )
        +
        int(
            real_clean[
                "mcs_topology_timeout"
            ].sum()
        )
        +
        int(
            syn_clean[
                "mcs_order_timeout"
            ].sum()
        )
        +
        int(
            syn_clean[
                "mcs_topology_timeout"
            ].sum()
        )
    )

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)

    print(
        "TOTAL REMAINING TIMEOUT FLAGS =",
        total_left
    )


if __name__ == "__main__":
    main()
