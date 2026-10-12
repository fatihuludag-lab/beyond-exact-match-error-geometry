from pathlib import Path
import pickle
from rdkit import Chem

OLD = Path(
    r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_PARTIAL128\preds"
    r"\MSG_EG358_CONFIRM_DIFFMS_K1_rank_0_pred_0.pkl"
)

NEW = Path(
    r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_nw0\preds"
    r"\MSG_EG358_CONFIRM_DIFFMS_K1_ALL358_pred.pkl"
)

OLD_T = Path(
    r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_PARTIAL128\preds"
    r"\MSG_EG358_CONFIRM_DIFFMS_K1_rank_0_true_0.pkl"
)

NEW_T = Path(
    r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_nw0\preds"
    r"\MSG_EG358_CONFIRM_DIFFMS_K1_ALL358_true.pkl"
)


def graph_signature(m):
    atoms = tuple(
        (
            a.GetAtomicNum(),
            a.GetFormalCharge(),
            bool(a.GetIsAromatic()),
        )
        for a in m.GetAtoms()
    )

    bonds = tuple(sorted(
        (
            min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
            max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
            str(b.GetBondType()),
            bool(b.GetIsAromatic()),
        )
        for b in m.GetBonds()
    ))

    return atoms, bonds


with OLD.open("rb") as f:
    old = pickle.load(f)

with NEW.open("rb") as f:
    new = pickle.load(f)

with OLD_T.open("rb") as f:
    old_t = pickle.load(f)

with NEW_T.open("rb") as f:
    new_t = pickle.load(f)


assert len(old) == 128
assert len(new) == 358
assert len(old_t) == 128
assert len(new_t) == 358

pred_same = sum(
    graph_signature(old[i][0])
    ==
    graph_signature(new[i][0])
    for i in range(128)
)

truth_same = sum(
    graph_signature(old_t[i])
    ==
    graph_signature(new_t[i])
    for i in range(128)
)

print("=" * 80)
print("DIFFMS EG358 WORKER DETERMINISM AUDIT")
print("=" * 80)
print("TRUTH SAME =", truth_same, "/ 128")
print("PRED SAME  =", pred_same, "/ 128")

assert truth_same == 128

if pred_same == 128:
    print("DETERMINISM PASS")
else:
    print("DETERMINISM WARNING")
