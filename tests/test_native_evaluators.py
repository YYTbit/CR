import json
import subprocess

import pytest

from cr.native_evaluators import DuETEvaluator, RebuttalEvaluator, source_functions


class Backend:
    data_kind = "software_fixture"
    identity = {"kind": "software_fixture"}

    def __init__(self, responses):
        self.responses = iter(responses)

    def complete(self, messages, **options):
        return {"text": next(self.responses), "usage": {"prompt_tokens": 3, "completion_tokens": 1}}

    def close(self):
        pass


def repository(tmp_path, files):
    for name, text in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Anonymous", "-c", "user.email=anonymous@example.invalid",
                    "-c", "commit.gpgsign=false", "commit", "-qm", "Fixture source"], cwd=tmp_path, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()


def options(tmp_path, revision, backend):
    return {"source": str(tmp_path), "expected_revision": revision, "backend": backend,
            "schema": {"identifier": "id", "context_fields": ["question"], "state_label": "state",
                       "positive_labels": {"0": 0, "1": 1}}}


def test_source_loader_skips_script_execution_and_supports_native_methods(tmp_path):
    path = tmp_path / "source.py"
    path.write_text('PRICE = {"input": 5 / 1000000}\n'
                    'raise RuntimeError("top level script is not a scorer")\n'
                    'class Prompts:\n'
                    '    def render(self, value: MissingAnnotation):\n'
                    '        return "Task " + value\n')
    loaded = source_functions(path, ["Prompts.render"], ["PRICE"])
    assert loaded.render(None, "question") == "Task question"
    assert loaded.PRICE["input"] == pytest.approx(5e-6)


def test_duet_uses_three_persuasion_exchanges_and_separate_stance_checks(tmp_path):
    revision = repository(tmp_path, {
        "prompts.py": 'def ask_mcq_question(question, options):\n    return question + " Options " + str(options)\n'
                      'def build_persuasion_prompt(letter, answer, appeal):\n    return letter + " " + answer + " " + appeal\n',
        "utils.py": 'def parse_single_letter_answer(content):\n    return content if len(content) == 1 else None\n'})
    evaluator = DuETEvaluator(**options(tmp_path, revision, Backend(["A", "reply", "B", "reply", "A", "reply", "B"])))
    record = {"id": "item", "question": "Question", "options": ["correct", "target"],
              "answer_idx": 0, "target_idx": 1, "state": 1, "domain": "fixture"}
    row = evaluator.score(record, "candidate appeal")
    assert row["initial_correct"] is True and row["turn_three_correct"] is False
    assert len(row["native_trace"]) == 7
    assert [turn["correct"] for turn in row["attempts"]] == [False, True, False]
    assert all("candidate appeal" in row["native_trace"][index]["messages"][-1]["content"] for index in (1, 3, 5))
    assert row["data_kind"] == "software_fixture"
    assert evaluator.sender_context(record) == "question\nQuestion"
    assert evaluator.sender_state(record) == 1
    assert evaluator.proposition(record) == "item"


def test_native_sources_are_verified_against_their_commit(tmp_path):
    revision = repository(tmp_path, {"prompts.py": 'def ask_mcq_question(a, b):\n    return a\n'
                                    'def build_persuasion_prompt(a, b, c):\n    return c\n',
                                     "utils.py": 'def parse_single_letter_answer(content):\n    return content\n'})
    (tmp_path / "utils.py").write_text('def parse_single_letter_answer(content):\n    return "A"\n')
    with pytest.raises(ValueError, match="contains changes"):
        DuETEvaluator(**options(tmp_path, revision, Backend([])))


def test_rebuttal_preserves_four_dimensions_and_candidate_evidence(tmp_path):
    revision = repository(tmp_path, {"eval/agent_prompt.py": 'SYS4 = "Native fixture grading contract"\n',
                                     "eval/ag_eval_ours.py": 'raise RuntimeError("script entrypoint")\n'
                                     'def RW_input1(record):\n    return record["evidence"] + " " + record["response"]\n'})
    dimensions = {"Attitude": 2, "Clarity": 4, "Persuasiveness": 6, "Constructiveness": 8}
    evaluator = RebuttalEvaluator(**options(tmp_path, revision, Backend([json.dumps({"score": dimensions})])))
    row = evaluator.score({"evidence": "paper evidence"}, "candidate rebuttal")
    assert row["dimensions"] == dimensions
    assert row["rebuttal_rm_score"] == 5
    assert row["native_trace"][0]["messages"][-1]["content"] == "paper evidence candidate rebuttal"


def test_invalid_native_letter_is_recorded_and_three_appeals_complete(tmp_path):
    revision = repository(tmp_path, {"prompts.py": 'def ask_mcq_question(a, b):\n    return a\n'
                                    'def build_persuasion_prompt(a, b, c):\n    return c\n',
                                     "utils.py": 'def parse_single_letter_answer(content):\n    return None\n'})
    evaluator = DuETEvaluator(**options(tmp_path, revision, Backend(["unparseable"] * 7)))
    row = evaluator.score({"question": "Question", "options": ["A", "B"], "answer_idx": 0}, "appeal")
    assert row["initial_correct"] is False and row["initial_valid_char"] is False
    assert len(row["attempts"]) == 3
    assert all(not attempt["valid_char"] and not attempt["correct"] for attempt in row["attempts"])
