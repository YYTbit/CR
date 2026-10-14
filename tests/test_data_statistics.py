import numpy as np
import pytest

from cr.data import prepare_records, proposition_split, ternary_action
from cr.statistics import holm_correction, paired_comparison, resolution_budget, seed_summary


def test_resource_prior_is_derived_from_labels_and_context_preserves_native_fields():
    rows = [{"id": str(index), "message": "message", "query": "query", "label": label, "topic": "topic" + str(index // 2)}
            for index, label in enumerate([0, 1, 1, 1])]
    schema = {"identifier": "id", "message": "message", "context_fields": ["query"], "state_label": "label",
              "positive_labels": {"0": 0, "1": 1}, "proposition": "topic", "split": "released"}
    game, messages, metadata = prepare_records(rows, schema, "EoBench")
    np.testing.assert_allclose(game.prior, [.25, .75])
    assert all("label" not in message.context for message in messages)
    first, second = proposition_split(messages, .5, 2)
    assert not {message.proposition for message in first} & {message.proposition for message in second}
    assert ternary_action(1, 1, 5) == 0 and ternary_action(5, 1, 5) == 2 and ternary_action(3, 1, 5) == 1


def test_statistics_use_sample_seed_deviation_paired_seeds_and_holm_family():
    summary = seed_summary([1, 2, 3])
    assert summary["mean"] == 2 and summary["std"] == 1
    result = paired_comparison({0: 2, 1: 4, 2: 5}, {0: 1, 1: 2, 2: 3})
    assert result["degrees_of_freedom"] == 2
    correction = holm_correction({"a": .001, "b": .02, "c": .04})
    assert correction["a"]["holm_p"] == pytest.approx(.003)
    assert correction["b"]["holm_p"] == pytest.approx(.04)
    assert correction["c"]["holm_p"] == pytest.approx(.04)
    with pytest.raises(ValueError):
        paired_comparison({0: 1, 1: 2}, {1: 2, 2: 3})
    assert resolution_budget(.01, .1)["samples_total"] == 4
    assert resolution_budget(.01, .1)["confidence_level"] is None
