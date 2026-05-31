from __future__ import annotations

import sqlite3

import pytest

from loom_agent import (
    CheckpointPolicy,
    Continue,
    RunContext,
    StepExecutionError,
    ErrorCategory,
)

from tests.test_runner_resume import State, counter_step, decode_result, decode_state, encode_result, encode_state, runner


@pytest.mark.asyncio
async def test_compact_skips_successful_attempt_rows(tmp_path) -> None:
    agent = runner(tmp_path, checkpoint_policy=CheckpointPolicy(mode="compact"))
    first = await agent.start(run_id="compact-ok", initial_state=State(0, 50), max_steps=10)
    assert first.status == "paused"
    finished = await agent.resume(run_id="compact-ok", max_steps=100)
    assert finished.status == "completed"
    assert agent.store.count_attempts("compact-ok") == 0
    assert agent.store.count_steps("compact-ok") > 0


@pytest.mark.asyncio
async def test_compact_retains_failed_attempt(tmp_path) -> None:
    async def fail_step(state: State, ctx: RunContext) -> Continue[State]:
        raise RuntimeError("boom")

    agent = runner(tmp_path, step=fail_step, checkpoint_policy=CheckpointPolicy(mode="compact"))
    with pytest.raises(RuntimeError, match="boom"):
        await agent.start(run_id="compact-fail", initial_state=State(0, 3), max_steps=1)

    attempts = agent.get_attempts("compact-fail", step_index=0)
    assert len(attempts) == 1
    assert attempts[0].status == "failed"


@pytest.mark.asyncio
async def test_compact_retains_retry_attempts(tmp_path) -> None:
    calls = 0

    async def flaky_step(state: State, ctx: RunContext) -> Continue[State]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StepExecutionError("wobble", error_category=ErrorCategory.TRANSIENT)
        return Continue(State(current=state.current + 1, target=state.target))

    agent = runner(tmp_path, step=flaky_step, checkpoint_policy=CheckpointPolicy(mode="compact"))
    await agent.start(run_id="compact-retry", initial_state=State(0, 5), max_steps=1)

    attempts = agent.get_attempts("compact-retry", step_index=0)
    assert [a.attempt_index for a in attempts] == [0, 1]
    assert [a.status for a in attempts] == ["failed", "completed"]


@pytest.mark.asyncio
async def test_compact_resume_idempotent(tmp_path) -> None:
    agent = runner(tmp_path, checkpoint_policy=CheckpointPolicy(mode="compact"))
    first = await agent.start(run_id="compact-resume", initial_state=State(0, 5), max_steps=3)
    assert first.status == "paused"
    second = await agent.resume(run_id="compact-resume", max_steps=10)
    assert second.status == "completed"
    assert agent.store.count_attempts("compact-resume") == 0


@pytest.mark.asyncio
async def test_compact_verify_run_valid(tmp_path) -> None:
    agent = runner(tmp_path, checkpoint_policy=CheckpointPolicy(mode="compact"))
    first = await agent.start(run_id="compact-verify", initial_state=State(0, 20), max_steps=5)
    assert first.status == "paused"
    finished = await agent.resume(run_id="compact-verify", max_steps=50)
    assert finished.status == "completed"
    verification = agent.verify_run("compact-verify")
    assert verification.valid is True


@pytest.mark.asyncio
async def test_compact_explain_notes_compact_attempt_log(tmp_path) -> None:
    agent = runner(tmp_path, checkpoint_policy=CheckpointPolicy(mode="compact"))
    first = await agent.start(run_id="compact-explain", initial_state=State(0, 10), max_steps=3)
    assert first.status == "paused"
    await agent.resume(run_id="compact-explain", max_steps=20)
    explanation = agent.explain_run("compact-explain")
    assert "compact_attempt_log" in explanation.invariant_notes


def test_schema_drops_redundant_indexes(tmp_path) -> None:
    from loom_agent import SQLiteCheckpointStore

    db_path = tmp_path / "schema.sqlite"
    store = SQLiteCheckpointStore(db_path)
    store.close()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "select name from sqlite_master where type='index' and name not like 'sqlite_%'"
        ).fetchall()
        index_names = {str(row[0]) for row in rows}
    assert "idx_steps_run_step" not in index_names
    assert "idx_attempts_run_step_attempt" not in index_names
    assert "idx_tool_calls_idempotency_key" in index_names
    version = int(sqlite3.connect(db_path).execute("pragma user_version").fetchone()[0])
    assert version == 2
