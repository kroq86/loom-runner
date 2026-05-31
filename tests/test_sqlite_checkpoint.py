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
