from pathlib import Path
import pickle
import re

D = Path(
    r"C:\research\ms_error_geometry\results\diffms_eg358_confirm_k1_nw0\preds"
)

OUT_PRED = D / "MSG_EG358_CONFIRM_DIFFMS_K1_ALL358_pred.pkl"
OUT_TRUE = D / "MSG_EG358_CONFIRM_DIFFMS_K1_ALL358_true.pkl"


def suffix_number(path):
    m = re.search(r"_(\d+)\.pkl$", path.name)
    if not m:
        raise RuntimeError(path.name)
    return int(m.group(1))


pred_files = sorted(
    D.glob("MSG_EG358_CONFIRM_DIFFMS_K1_rank_0_pred_*.pkl"),
    key=suffix_number
)

true_files = sorted(
    D.glob("MSG_EG358_CONFIRM_DIFFMS_K1_rank_0_true_*.pkl"),
    key=suffix_number
)

print("PRED FILES")
for f in pred_files:
    print(" ", f.name)

print()
print("TRUE FILES")
for f in true_files:
    print(" ", f.name)


pred = []
truth = []

for f in pred_files:
    with open(f, "rb") as fh:
        x = pickle.load(fh)

    print(f.name, "=", len(x))
    pred.extend(x)

for f in true_files:
    with open(f, "rb") as fh:
        x = pickle.load(fh)

    print(f.name, "=", len(x))
    truth.extend(x)


print()
print("MERGED PRED =", len(pred))
print("MERGED TRUE =", len(truth))
print(
    "NONEMPTY PRED =",
    sum(bool(x) for x in pred)
)

assert len(pred) == 358
assert len(truth) == 358
assert all(len(x) == 1 for x in pred)


with open(OUT_PRED, "wb") as fh:
    pickle.dump(pred, fh)

with open(OUT_TRUE, "wb") as fh:
    pickle.dump(truth, fh)


print()
print("MERGE PASS")
print("PRED =", OUT_PRED)
print("TRUE =", OUT_TRUE)
