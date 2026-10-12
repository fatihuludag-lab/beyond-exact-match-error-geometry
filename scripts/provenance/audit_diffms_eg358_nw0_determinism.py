import pickle
from rdkit import Chem

OLD_PRED = r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_PARTIAL128\preds\MSG_EG358_CONFIRM_DIFFMS_K1_rank_0_pred_0.pkl"
OLD_TRUE = r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_PARTIAL128\preds\MSG_EG358_CONFIRM_DIFFMS_K1_rank_0_true_0.pkl"

NEW_PRED = r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_nw0\preds\MSG_EG358_CONFIRM_DIFFMS_K1_rank_0_pred_0.pkl"
NEW_TRUE = r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_nw0\preds\MSG_EG358_CONFIRM_DIFFMS_K1_rank_0_true_0.pkl"


def graph_smiles(m):
    if m is None:
        return None

    try:
        x = Chem.Mol(m)
        Chem.RemoveStereochemistry(x)

        return Chem.MolToSmiles(
            x,
            canonical=True,
            isomericSmiles=False
        )
    except Exception as e:
        return f"ERROR:{type(e).__name__}"


with open(OLD_PRED, "rb") as f:
    op = pickle.load(f)

with open(OLD_TRUE, "rb") as f:
    ot = pickle.load(f)

with open(NEW_PRED, "rb") as f:
    np = pickle.load(f)

with open(NEW_TRUE, "rb") as f:
    nt = pickle.load(f)


print("OLD PRED =", len(op))
print("OLD TRUE =", len(ot))
print("NEW PRED =", len(np))
print("NEW TRUE =", len(nt))

assert len(op) == 128
assert len(ot) == 128
assert len(np) == 358
assert len(nt) == 358


pred_same = 0
truth_same = 0
pred_diff = []

for i in range(128):

    a = graph_smiles(op[i][0])
    b = graph_smiles(np[i][0])

    if a == b:
        pred_same += 1
    else:
        pred_diff.append((i, a, b))

    ta = graph_smiles(ot[i])
    tb = graph_smiles(nt[i])

    if ta == tb:
        truth_same += 1


print()
print("=" * 80)
print("DETERMINISM AUDIT")
print("=" * 80)

print(
    "TRUTH SAME FIRST128 =",
    truth_same,
    "/ 128"
)

print(
    "PRED SAME FIRST128 =",
    pred_same,
    "/ 128"
)

print(
    "PRED DIFFERENT =",
    len(pred_diff)
)

if pred_diff:
    print()
    print("FIRST DIFFERENCES")

    for x in pred_diff[:10]:
        print(x)

if truth_same != 128:
    raise RuntimeError(
        "Truth ordering changed!"
    )

print()
print(
    "ORDERING PASS"
)

if pred_same == 128:
    print(
        "DETERMINISM PASS: num_workers change did not alter predictions."
    )
else:
    print(
        "WARNING: prediction sequence changed."
    )
