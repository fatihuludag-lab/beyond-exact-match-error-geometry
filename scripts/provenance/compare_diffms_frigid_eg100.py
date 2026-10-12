from pathlib import Path
import pandas as pd
import numpy as np

D_PATH = Path(
    r"results\normalized\DIFFMS_K1_EG100.csv"
)

F_PATH = Path(
    r"results\normalized\FRIGID_BASE_FM1_EG100_HARMONIZED.csv"
)

OUT = Path(
    r"results\paired_DIFFMS_vs_FRIGID_EG100.csv"
)

d = pd.read_csv(D_PATH)
f = pd.read_csv(F_PATH)

d = d.rename(columns={
    "task_valid": "diffms_task_valid",
    "same_formula": "diffms_same_formula",
    "exact_connectivity": "diffms_exact",
    "tanimoto": "diffms_tanimoto",
})

f = f.rename(columns={
    "task_valid": "frigid_task_valid",
    "same_formula": "frigid_same_formula",
    "exact_connectivity": "frigid_exact",
    "tanimoto": "frigid_tanimoto",
})

keep_d = [
    "panel_index",
    "spec",
    "diffms_task_valid",
    "diffms_same_formula",
    "diffms_exact",
    "diffms_tanimoto",
]

keep_f = [
    "panel_index",
    "spec",
    "frigid_task_valid",
    "frigid_same_formula",
    "frigid_exact",
    "frigid_tanimoto",
]

x = d[keep_d].merge(
    f[keep_f],
    on=["panel_index", "spec"],
    how="inner",
    validate="one_to_one",
)

assert len(x) == 100

x["both_task_valid"] = (
    x["diffms_task_valid"]
    &
    x["frigid_task_valid"]
)

x["both_wrong"] = (
    ~x["diffms_exact"]
    &
    ~x["frigid_exact"]
)

x["common_valid_wrong"] = (
    x["both_task_valid"]
    &
    x["both_wrong"]
)

x["delta_tanimoto_F_minus_D"] = (
    x["frigid_tanimoto"]
    -
    x["diffms_tanimoto"]
)

cvw = x[x["common_valid_wrong"]].copy()

print("=" * 100)
print("DIFFMS-K1 vs FRIGID-base@FM1 — MSG-EG100")
print("=" * 100)

print("N =", len(x))

print()
print("DIFFMS")
print(" task-valid =", int(x["diffms_task_valid"].sum()))
print(" same-formula =", int(x["diffms_same_formula"].sum()))
print(" exact =", int(x["diffms_exact"].sum()))

print()
print("FRIGID")
print(" task-valid =", int(x["frigid_task_valid"].sum()))
print(" same-formula =", int(x["frigid_same_formula"].sum()))
print(" exact =", int(x["frigid_exact"].sum()))

print()
print("PAIRED SETS")
print(" both task-valid =", int(x["both_task_valid"].sum()))
print(" both wrong =", int(x["both_wrong"].sum()))
print(" common valid wrong =", len(cvw))

if len(cvw):
    print()
    print("COMMON-VALID-WRONG TANIMOTO")
    print(
        "DiffMS mean =",
        cvw["diffms_tanimoto"].mean()
    )
    print(
        "FRIGID mean =",
        cvw["frigid_tanimoto"].mean()
    )
    print(
        "DiffMS median =",
        cvw["diffms_tanimoto"].median()
    )
    print(
        "FRIGID median =",
        cvw["frigid_tanimoto"].median()
    )

    delta = cvw["delta_tanimoto_F_minus_D"]

    print(
        "Mean delta F-D =",
        delta.mean()
    )
    print(
        "Median delta F-D =",
        delta.median()
    )

    print(
        "FRIGID better =",
        int((delta > 0).sum()),
        "/",
        len(cvw)
    )

    print(
        "DiffMS better =",
        int((delta < 0).sum()),
        "/",
        len(cvw)
    )

    print(
        "Tie =",
        int((delta == 0).sum()),
        "/",
        len(cvw)
    )

x.to_csv(OUT, index=False)

print()
print("Written:", OUT)
