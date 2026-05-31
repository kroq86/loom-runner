from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from flow_xray import trace

from loom_agent import AgentRunner, Complete, Continue, RunContext, SQLiteCheckpointStore


@dataclass(frozen=True)
class State:
    current: int
    target: int


def encode_state(state: State) -> dict[str, int]:
    return {"current": state.current, "target": state.target}


def decode_state(data: Any) -> State:
    return State(current=int(data["current"]), target=int(data["target"]))


def encode_result(result: dict[str, int]) -> dict[str, int]:
    return result


def decode_result(data: Any) -> dict[str, int]:
    return {"current": int(data["current"])}


async def counter_step(state: State, ctx: RunContext) -> Continue[State] | Complete[dict[str, int]]:
    await asyncio.sleep(0)
    if state.current >= state.target:
        return Complete({"current": state.current, "step": ctx.step_index})
    return Continue(State(current=state.current + 1, target=state.target))


def runner(tmp_path, step=counter_step) -> AgentRunner[State, dict[str, int]]:
    return AgentRunner(
        step=step,
        store=SQLiteCheckpointStore(tmp_path / "runs.sqlite"),
        encode_state=encode_state,
        decode_state=decode_state,
        encode_result=encode_result,
        decode_result=decode_result,
    )


@pytest.mark.asyncio
async def test_start_pause_resume_complete(tmp_path) -> None:
    agent = runner(tmp_path)

    first = await agent.start(run_id="demo", initial_state=State(0, 10), max_steps=5)
    second = await agent.resume(run_id="demo", max_steps=100)

    assert first.status == "paused"
    assert first.step_index == 5
    assert first.state == State(5, 10)
    assert second.status == "completed"
    assert second.step_index == 10
    assert second.result == {"current": 10, "step": 10}


@pytest.mark.asyncio
async def test_completed_run_cannot_resume(tmp_path) -> None:
    agent = runner(tmp_path)

    result = await agent.start(run_id="demo", initial_state=State(0, 1), max_steps=10)
    assert result.status == "completed"

    with pytest.raises(ValueError, match="run already completed: demo"):
        await agent.resume(run_id="demo", max_steps=1)


@pytest.mark.asyncio
async def test_duplicate_start_fails(tmp_path) -> None:
    agent = runner(tmp_path)

    await agent.start(run_id="demo", initial_state=State(0, 10), max_steps=1)

    with pytest.raises(ValueError, match="run already exists: demo"):
        await agent.start(run_id="demo", initial_state=State(0, 10), max_steps=1)


@pytest.mark.asyncio
async def test_failed_step_marks_run_failed_and_preserves_error(tmp_path) -> None:
    async def fail_step(state: State, ctx: RunContext) -> Continue[State]:
        raise RuntimeError(f"boom at {ctx.step_index}")

    agent = runner(tmp_path, step=fail_step)

    with pytest.raises(RuntimeError, match="boom at 0"):
        await agent.start(run_id="demo", initial_state=State(0, 10), max_steps=1)

    stored = agent.store.get_run("demo")
    assert stored.status == "failed"
    assert stored.error == "RuntimeError: boom at 0"


@pytest.mark.asyncio
async def test_invalid_step_result_marks_run_failed(tmp_path) -> None:
    async def bad_step(state: State, ctx: RunContext):
        return {"not": "an outcome"}

    agent = runner(tmp_path, step=bad_step)

    with pytest.raises(TypeError, match="step must return Continue or Complete"):
        await agent.start(run_id="demo", initial_state=State(0, 10), max_steps=1)

    stored = agent.store.get_run("demo")
    assert stored.status == "failed"
    assert stored.error == "TypeError: step must return Continue or Complete, got dict"


@pytest.mark.asyncio
async def test_runner_handles_10k_steps_without_recursion_failure(tmp_path) -> None:
    agent = runner(tmp_path)

    result = await agent.start(run_id="deep", initial_state=State(0, 10_000), max_steps=20_000)

    assert result.status == "completed"
    assert result.step_index == 10_000


@pytest.mark.asyncio
async def test_concurrent_runs_do_not_leak_state(tmp_path) -> None:
    agent = runner(tmp_path)

    results = await asyncio.gather(
        *(
            agent.start(
                run_id=f"run-{index}",
                initial_state=State(0, 100 + index),
                max_steps=200,
            )
            for index in range(4)
        )
    )

    assert [result.status for result in results] == ["completed"] * 4
    assert [result.result for result in results] == [
        {"current": 100, "step": 100},
        {"current": 101, "step": 101},
        {"current": 102, "step": 102},
        {"current": 103, "step": 103},
    ]


def test_trace_html_smoke_for_runner_steps(tmp_path) -> None:
    agent = runner(tmp_path)
    trace_path = tmp_path / "trace.html"

    trace_result = trace.run(
        lambda: asyncio.run(
            agent.start(run_id="traced", initial_state=State(0, 3), max_steps=10)
        )
    )
    trace_result.to_html(str(trace_path))

    assert trace_result.return_value.status == "completed"
    html = trace_path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "_step_once" in html
