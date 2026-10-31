"""Source-field fault injection; real downloads stay in ignored audit artifacts."""

import copy
import json

import pytest

from nice_validation.data import main_task_records
from nice_validation.schemas import task_record, training_record


def smith():
    return {
        "instance_id": "oauthlib__oauthlib.1fd52536.combine_file__example",
        "repo": "swesmith/oauthlib__oauthlib.1fd52536",
        "problem_statement": "Scope conversion reverses the order.",
        "image_name": "example/swesmith:fixture",
        "FAIL_TO_PASS": ["tests/test_scope.py::test_roundtrip"],
        "PASS_TO_PASS": [],
    }


@pytest.mark.parametrize("value", [None, "", " \n\t"])
def test_blank_smith_issue_is_excluded_before_sampling_without_fabricating_text(value):
    good = smith()
    absent = {**smith(), "instance_id": "empty-issue", "problem_statement": value}
    original = copy.deepcopy([good, absent])
    rows, report = main_task_records([good, absent], "swesmith")
    assert rows == [good]
    assert report["raw_read"] == 2
    assert report["excluded_task_ids"] == ["empty-issue"]
    assert [good, absent] == original


@pytest.mark.parametrize(
    "field",
    [
        "instance_id",
        "repo",
        "problem_statement",
        "image_name",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    ],
)
@pytest.mark.parametrize("fault", ["missing", "null", "empty"])
def test_smith_required_fields_fail_at_the_source_boundary(field, fault):
    row = smith()
    if fault == "missing":
        row.pop(field)
    else:
        row[field] = None if fault == "null" else ""
    with pytest.raises(ValueError, match=field):
        task_record(row, "swesmith")


def test_harness_test_lists_support_real_json_strings_without_mutating_source():
    row = smith()
    row["FAIL_TO_PASS"] = json.dumps(row["FAIL_TO_PASS"])
    row["PASS_TO_PASS"] = "[]"
    result = task_record(row, "swesmith")
    assert result["FAIL_TO_PASS"] == ["tests/test_scope.py::test_roundtrip"]
    assert result["PASS_TO_PASS"] == []
    assert isinstance(row["FAIL_TO_PASS"], str)


@pytest.mark.parametrize("backend", ["swebench", "live"])
@pytest.mark.parametrize("field", ["base_commit", "FAIL_TO_PASS", "problem_statement"])
def test_benchmark_fields_cannot_be_silently_defaulted(backend, field):
    row = {
        **smith(),
        "base_commit": "a" * 40,
        "version": "4.3",
        "test_cmds": ["pytest"],
        "log_parser": "pytest",
    }
    row.pop(field)
    with pytest.raises(ValueError, match=field):
        task_record(row, backend)


def test_live_native_schema_does_not_invent_swebench_repository_version():
    row = {
        **smith(),
        "base_commit": "a" * 40,
        "test_cmds": ["pytest"],
        "log_parser": "pytest",
    }
    assert "version" not in task_record(row, "live")
    with pytest.raises(ValueError, match="version"):
        task_record(row, "swebench")
    for field in ["test_cmds", "log_parser"]:
        bad = dict(row)
        bad.pop(field)
        with pytest.raises(ValueError, match=field):
            task_record(bad, "live")


@pytest.mark.parametrize("field", ["instance_id", "traj_id", "messages", "resolved"])
def test_training_source_missing_fields_are_reported(field):
    row = {
        "instance_id": "owner__repo.task",
        "traj_id": "trace-1",
        "messages": [],
        "resolved": True,
    }
    row.pop(field)
    with pytest.raises(ValueError, match="resolved" if field == "resolved" else field):
        training_record(row)


@pytest.mark.parametrize("value", ["../outside", "..", "a/b", "a\\b"])
def test_task_ids_cannot_escape_episode_evidence_directory(value):
    with pytest.raises(ValueError, match="path_component"):
        task_record({**smith(), "instance_id": value}, "swesmith")
