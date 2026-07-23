import pandas as pd

from scripts.analyze_e5_transfer_behavior import INPUT_CSV, build_summary


def test_formal_e5_transfer_behavior_summary():
    summary = build_summary(pd.read_csv(INPUT_CSV))

    assert summary["budget"].tolist() == [50, 100, 200]
    assert (summary["n_run_pairs"] == 500).all()
    assert (summary["n_instance_blocks"] == 50).all()
    assert (summary["online_policy_entropy_median"] < 0.7).all()
    assert (summary["scratch_policy_entropy_median"] > 2.29).all()
    assert (summary["online_uniform_ma_share_median"] > 0.3).all()
    assert (summary["scratch_uniform_ma_share_median"] < 0.12).all()
    assert (
        summary.loc[summary["budget"] == 200, "js_divergence_vs_scratch_median"].iloc[0]
        > 0.19
    )
