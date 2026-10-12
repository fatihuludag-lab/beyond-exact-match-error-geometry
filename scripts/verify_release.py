from pathlib import Path
import pandas as pd

def scientific_verify(root: Path):
    print("=" * 96)
    print("SCIENTIFIC RELEASE VERIFICATION")
    print("=" * 96)

    panels = {
        "MSG_EG100_DEV_v01.csv": 100,
        "MSG_EG500_PREFREEZE_v01.csv": 500,
        "MSG_EG400_CONFIRM_v01.csv": 400,
        "MSG_EG358_COMMON_SUPPORT_CONFIRM_v01.csv": 358,
    }
    for fn, n in panels.items():
        d = pd.read_csv(root / "manifests" / fn)
        assert len(d) == n, (fn, len(d), n)
        print(f"PASS panel {fn}: {n}")

    expected = {
        "DIFFMS_K1_EG358_CONFIRM.csv": (358, 158, 38, 0),
        "FRIGID_BASE_FM1_EG358_CONFIRM_HARMONIZED.csv": (358, 358, 333, 10),
        "MS2FORGE_K1_EG358_CONFIRM.csv": (358, 226, 108, 39),
    }

    for fn, (n, valid, formula, exact) in expected.items():
        d = pd.read_csv(root / "data" / "normalized_predictions" / fn)
        assert len(d) == n
        assert int(as_bool(d["task_valid"]).sum()) == valid
        assert int(as_bool(d["same_formula"]).sum()) == formula
        assert int(as_bool(d["exact_connectivity"]).sum()) == exact
        print(f"PASS {fn}: valid={valid}, formula={formula}, exact={exact}")

    rq2 = pd.read_csv(
        root / "data" / "derived_analysis_tables" /
        "EG358_CONFIRM_RQ2_MCES_FINAL_MANIFEST_TRUTH.csv"
    )
    assert len(rq2) == 3

    rq2_expected = {
        "DiffMS_vs_FRIGID": (-3.4903846153846154, -5.1506410256410255, -1.8365384615384615),
        "DiffMS_vs_MS2Forge": (-6.101063829787234, -7.659574468085107, -4.6063829787234045),
        "FRIGID_vs_MS2Forge": (-4.0467032967032965, -5.406593406593407, -2.681318681318681),
    }

    for name, vals in rq2_expected.items():
        r = rq2.loc[rq2["comparison"] == name].iloc[0]
        close(r["mean_delta_B_minus_A"], vals[0])
        close(r["mean_ci_low"], vals[1])
        close(r["mean_ci_high"], vals[2])
        print(f"PASS RQ2 {name}")

    rq2_mces = pd.read_csv(
        root / "mces" / "rq2" /
        "EG358_CONFIRM_RQ2_MCES_OUT_MANIFEST_TRUTH.csv",
        header=None,
        names=["pair_id", "mces", "time", "mode"],
    )
    assert len(rq2_mces) == 585
    assert (rq2_mces["mces"] >= 0).all()
    assert set(rq2_mces["mode"]) == {1}
    print("PASS RQ2 exact MCES: 585 / 585")

    parent = pd.read_csv(
        root / "data" / "derived_analysis_tables" /
        "RQ4_CONFIRM_FINAL_PARENT_MATCHED_EG358.csv"
    )
    assert len(parent) == 333
    close(parent["real_median_mces"].median(), 17.0)
    close(parent["synthetic_median_mces"].median(), 2.0)

    delta = parent["delta_mces_real_minus_synthetic"]
    close(delta.mean(), 14.816816816816816)
    assert int((delta > 0).sum()) == 324
    assert int((delta < 0).sum()) == 1
    assert int((delta == 0).sum()) == 8
    print("PASS RQ4 parent matched: 333 parents; median MCES 17 vs 2")

    syn = pd.read_parquet(
        root / "data" / "synthetic_corruptions" /
        "RQ4_CONFIRM_SYNTHETIC_PRIMARY_EG358.parquet"
    )
    assert len(syn) == 3261
    assert syn["parent_spec"].nunique() == 333
    print("PASS synthetic corpus: 3261 observations / 333 parents")

    rq4_syn = pd.read_csv(
        root / "mces" / "rq4" / "RQ4_CONFIRM_SYNTHETIC_OUT.csv",
        header=None,
        names=["pair_id", "mces", "time", "mode"],
    )
    assert len(rq4_syn) == 3261
    assert (rq4_syn["mces"] >= 0).all()
    assert set(rq4_syn["mode"]) == {1}
    print("PASS RQ4 synthetic MCES: 3261 / 3261")

    rq4_new = pd.read_csv(
        root / "mces" / "rq4" / "RQ4_CONFIRM_REAL_MISSING_FINAL_OUT.csv",
        header=None,
        names=["pair_id", "mces", "time", "mode"],
    )
    assert len(rq4_new) == 108
    assert (rq4_new["mces"] >= 0).all()
    assert set(rq4_new["mode"]) == {1}
    print("PASS RQ4 new real MCES: 108 / 108")

    reuse = pd.read_csv(
        root / "mces" / "rq4" / "RQ4_CONFIRM_REAL_REUSED_FINAL.csv"
    )
    assert len(reuse) == 322
    print("PASS RQ4 reused exact MCES: 322")

    print()
    print("ALL SCIENTIFIC RELEASE GATES PASSED.")



if __name__ == "__main__":
    scientific_verify(Path(__file__).resolve().parents[1])
