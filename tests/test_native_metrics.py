import pytest

from cr.native import native_summary


def measured(values):
    return [dict(value, item=index, data_kind="measured_run") for index, value in enumerate(values)]


def test_eob_full_file_denominator_and_split_completeness():
    rows = measured([{"context_following": True}, {"context_following": False}, {"context_following": True}])
    result = native_summary("EoBench", rows, 3)
    assert result["value"] == 2 / 3 and result["denominator"] == 3
    with pytest.raises(ValueError):
        native_summary("EoBench", rows, 4)
    with pytest.raises(ValueError):
        native_summary("EoBench", rows + [rows[0]])


def test_duet_negative_flip_denominator_and_domain_pooling():
    rows = measured([{"initial_correct": True, "turn_three_correct": False, "domain": "first"},
                     {"initial_correct": True, "turn_three_correct": True, "domain": "second"},
                     {"initial_correct": False, "turn_three_correct": False, "domain": "first"}])
    result = native_summary("DuET-PD", rows)
    assert result["value"] == .5 and result["denominator"] == 2
    assert result["domains"]["first"]["negative_flips"] == 1
    assert native_summary("DuET-PD", [rows[-1]])["value"] is None


def test_pmiyc_ratings_and_mean_final_agreement():
    rows = measured([{"final_agreement": 1}, {"final_agreement": 3}, {"final_agreement": 5}])
    assert native_summary("PMIYC", rows)["value"] == 3
    with pytest.raises(ValueError):
        native_summary("PMIYC", measured([{"final_agreement": 3.5}]))


def test_sycon_counts_real_judged_flips_and_keeps_scenarios_separate():
    pattern = [True, False, True, False, True]
    debate = measured([{"scenario": "Debate", "alignments": pattern}])
    result = native_summary("SYCON-Bench", debate)
    assert result["value"] == 4 and result["range"] == [0, 4]
    ethical = measured([{"scenario": "Ethical", "alignments": [True, True, False, False, False]}])
    assert native_summary("SYCON-Bench", ethical)["value"] == 2
    with pytest.raises(ValueError):
        native_summary("SYCON-Bench", debate + [dict(ethical[0], item=1)])
    with pytest.raises(ValueError):
        native_summary("SYCON-Bench", measured([{"scenario": "Debate", "alignments": pattern[:4]}]))


def test_rebuttal_uses_finite_native_reward_scores():
    assert native_summary("RebuttalBench", measured([{"rebuttal_rm_score": 2}, {"rebuttal_rm_score": 3}]))["value"] == 2.5
    with pytest.raises(ValueError):
        native_summary("RebuttalBench", measured([{"rebuttal_rm_score": float("nan")}]))
