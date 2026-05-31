from __future__ import annotations

import sqlite3

import pytest

from loom_agent.checkpoint import SQLiteCheckpointStore


def test_sqlite_store_persists_across_instances(tmp_path) -> None:
    path = tmp_path / "runs.sqlite"
    store = SQLiteCheckpointStore(path)
    store.create_run(run_id="r1", state_json='{"n": 0}')
    store.save_step(run_id="r1", step_index=1, state_json='{"n": 1}')

    loaded = SQLiteCheckpointStore(path).get_run("r1")

    assert loaded.run_id == "r1"
    assert loaded.status == "running"
    assert loaded.step_index == 1
    assert loaded.state_json == '{"n": 1}'


def test_sqlite_store_duplicate_run_fails(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runs.sqlite")
    store.create_run(run_id="r1", state_json='{"n": 0}')

    with pytest.raises(ValueError, match="run already exists: r1"):
        store.create_run(run_id="r1", state_json='{"n": 0}')


def test_sqlite_store_steps_primary_key(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runs.sqlite")
    store.create_run(run_id="r1", state_json='{"n": 0}')
    store.save_step(run_id="r1", step_index=1, state_json='{"n": 1}')

    with pytest.raises(sqlite3.IntegrityError):
        store.save_step(run_id="r1", step_index=1, state_json='{"n": 1}')


def test_sqlite_store_lists_runs_and_steps(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runs.sqlite")
    store.create_run(run_id="r1", state_json='{"n": 0}')
    store.save_step(run_id="r1", step_index=1, state_json='{"n": 1}')
    store.create_run(run_id="r2", state_json='{"n": 10}')

    runs = store.list_runs()
    steps = store.get_steps("r1")

    assert [run.run_id for run in runs] == ["r1", "r2"]
    assert [step.step_index for step in steps] == [0, 1]
    assert [step.state_json for step in steps] == ['{"n": 0}', '{"n": 1}']


def test_sqlite_store_paginates_steps(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runs.sqlite")
    store.create_run(run_id="r1", state_json='{"n": 0}')
    for index in range(1, 5):
        store.save_step(run_id="r1", step_index=index, state_json=f'{{"n": {index}}}')

    steps = store.get_steps("r1", limit=2, offset=2)

    assert [step.step_index for step in steps] == [2, 3]


def test_sqlite_store_records_attempts_and_committed_step(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runs.sqlite")
    store.create_run(run_id="r1", state_json='{"n": 0}')

    store.record_attempt_started(
        run_id="r1",
        step_index=0,
        attempt_index=0,
        input_hash="abc",
    )
    store.record_attempt_finished(
        run_id="r1",
        step_index=0,
        attempt_index=0,
        status="completed",
        outcome_kind="continue",
        output_json='{"n": 1}',
        error=None,
        error_category=None,
        is_retryable=None,
    )
    store.commit_step_outcome(
        run_id="r1",
        step_index=0,
        input_hash="abc",
        outcome_kind="continue",
        state_json='{"n": 1}',
        result_json=None,
    )

    attempts = store.get_attempts("r1", step_index=0)
    committed = store.get_committed_step(run_id="r1", step_index=0, input_hash="abc")
    run = store.get_run("r1")

    assert [attempt.attempt_index for attempt in attempts] == [0]
    assert attempts[0].status == "completed"
    assert committed is not None
    assert committed.outcome_kind == "continue"
    assert committed.state_json == '{"n": 1}'
    assert run.step_index == 1
    assert run.state_json == '{"n": 1}'
    assert store.count_steps("r1") == 2
    assert store.count_attempts("r1") == 1
    assert store.count_retries("r1") == 0
    assert store.count_failed_attempts("r1") == 0
    assert store.last_failed_attempt("r1") is None
    assert store.has_step_gaps("r1") is False
    assert store.has_attempt_gaps("r1") is False


def test_sqlite_store_records_completed_tool_call_by_idempotency_key(tmp_path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runs.sqlite")
    store.create_run(run_id="r1", state_json='{"n": 0}')

    store.record_tool_call_started(
        tool_call_id="tool-1",
        run_id="r1",
        step_index=0,
        attempt_index=0,
        idempotency_key="same-call",
        tool_name="lookup",
        request_hash="req",
    )
    store.record_tool_call_finished(
        tool_call_id="tool-1",
        status="completed",
        result_json='{"ok": true}',
        error=None,
        error_category=None,
    )

    calls = store.get_tool_calls("r1", step_index=0)
    call = store.get_tool_call_by_key("same-call")

    assert len(calls) == 1
    assert call is not None
    assert call.tool_call_id == "tool-1"
    assert call.result_json == '{"ok": true}'
