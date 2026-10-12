from pathlib import Path
import csv
import pandas as pd
from rdkit import Chem

ROOT = Path(r"C:\research\ms_error_geometry")

REAL = ROOT / "results" / "RQ4_CONFIRM_REAL_THREE_MODELS_EG358.csv"

RQ2_MAP = (
    ROOT / "results" / "mces" /
    "EG358_CONFIRM_RQ2_MCES_MAP_MANIFEST_TRUTH.csv"
)

RQ2_OUT = (
    ROOT / "results" / "mces" /
    "EG358_CONFIRM_RQ2_MCES_OUT_MANIFEST_TRUTH.csv"
)

OUT_REUSE = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_REUSED_FINAL.csv"
)

OUT_MISSING_MAP = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_MISSING_FINAL_MAP.csv"
)

OUT_MISSING_INPUT = (
    ROOT / "results" / "mces" /
    "RQ4_CONFIRM_REAL_MISSING_FINAL_INPUT.csv"
)

MODEL_ALIAS = {
    "DiffMS-K1": "DiffMS",
    "FRIGID-base@FM1": "FRIGID",
    "MS2Forge-K1": "MS2Forge",
}


def canonical(s):
    m = Chem.MolFromSmiles(str(s), sanitize=False)

    if m is None:
        raise RuntimeError(f"Invalid SMILES: {s}")

    Chem.SanitizeMol(m)
    Chem.RemoveStereochemistry(m)

    return Chem.MolToSmiles(
        m,
        canonical=True,
        isomericSmiles=False
    )


def conn_key(s):
    m = Chem.MolFromSmiles(str(s), sanitize=False)

    if m is None:
        return None

    Chem.SanitizeMol(m)
    Chem.RemoveStereochemistry(m)

    k = Chem.MolToInchiKey(m)

    return k[:14] if k else None


real = pd.read_csv(REAL)

rq2_map = pd.read_csv(RQ2_MAP)

rq2_out = pd.read_csv(
    RQ2_OUT,
    header=None,
    names=[
        "pair_id",
        "mces",
        "time_sec",
        "mode",
    ]
)

assert len(real) == 430
assert len(rq2_map) == 585
assert len(rq2_out) == 585
assert set(rq2_out["mode"]) == {1}

rq2 = rq2_map.merge(
    rq2_out,
    on="pair_id",
    validate="one_to_one"
)

assert not rq2.duplicated(
    ["spec", "model"]
).any()

lookup = rq2.set_index(
    ["spec", "model"]
)


reused = []
missing = []

not_in_rq2 = 0
truth_repr_bad = 0
pred_repr_bad = 0


for r in real.itertuples(index=False):

    model2 = MODEL_ALIAS[r.model]

    key = (
        str(r.spec),
        model2,
    )

    truth = canonical(
        r.truth_smiles
    )

    pred = canonical(
        r.pred_smiles
    )

    if key not in lookup.index:

        not_in_rq2 += 1

        missing.append({
            "model": r.model,
            "spec": r.spec,
            "parent_spec": r.spec,
            "truth_smiles": truth,
            "pred_smiles": pred,
            "reason": "not_in_RQ2",
        })

        continue


    old = lookup.loc[key]

    old_truth = canonical(
        old["truth_smiles"]
    )

    old_pred = canonical(
        old["pred_smiles"]
    )


    if conn_key(truth) != conn_key(old_truth):
        raise RuntimeError(
            f"Truth connectivity mismatch: {key}"
        )

    if conn_key(pred) != conn_key(old_pred):
        raise RuntimeError(
            f"Prediction connectivity mismatch: {key}"
        )


    if truth != old_truth:

        truth_repr_bad += 1

        missing.append({
            "model": r.model,
            "spec": r.spec,
            "parent_spec": r.spec,
            "truth_smiles": truth,
            "pred_smiles": pred,
            "reason": "truth_representation_changed",
        })

        continue


    if pred != old_pred:

        pred_repr_bad += 1

        missing.append({
            "model": r.model,
            "spec": r.spec,
            "parent_spec": r.spec,
            "truth_smiles": truth,
            "pred_smiles": pred,
            "reason": "prediction_representation_changed",
        })

        continue


    reused.append({
        "model": r.model,
        "spec": r.spec,
        "parent_spec": r.spec,
        "truth_smiles": truth,
        "pred_smiles": pred,
        "mces": float(old["mces"]),
        "mces_source": "RQ2_manifest_truth_exact_reuse",
    })


reused = pd.DataFrame(reused)
missing = pd.DataFrame(missing)


print("=" * 100)
print("RQ4 FINAL REAL MCES REUSE")
print("=" * 100)

print("REAL TOTAL =", len(real))
print("REUSED EXACT MCES =", len(reused))
print("NEW REAL MCES REQUIRED =", len(missing))

print()
print("not in RQ2 =", not_in_rq2)
print("truth representation changed =", truth_repr_bad)
print("prediction representation changed =", pred_repr_bad)

assert len(reused) + len(missing) == 430

# After authoritative RQ2 correction this should normally be zero.
if truth_repr_bad:
    print()
    print(
        "WARNING: authoritative truth representations still differ."
    )


reused.to_csv(
    OUT_REUSE,
    index=False
)


if len(missing):

    missing = missing.reset_index(drop=True)

    missing["pair_id"] = range(
        len(missing)
    )

    missing.to_csv(
        OUT_MISSING_MAP,
        index=False
    )

    with OUT_MISSING_INPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        for r in missing.itertuples(index=False):

            writer.writerow([
                r.pair_id,
                r.truth_smiles,
                r.pred_smiles,
            ])


print()
print("REUSE =", OUT_REUSE)
print("MISSING MAP =", OUT_MISSING_MAP)
print("MISSING INPUT =", OUT_MISSING_INPUT)

print()
print("RQ4 FINAL REUSE AUDIT PASS")
