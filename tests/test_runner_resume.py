from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any

import pytest
from flow_xray import trace

from loom_agent import (
    AgentRunner,
    CheckpointPolicy,
    Complete,
    Continue,
    ErrorCategory,
    PayloadPolicy,
    RunContext,
    SQLiteCheckpointStore,
    StepExecutionError,
    ToolRegistry,
)


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
    result = {"current": int(data["current"])}
    if "step" in data:
        result["step"] = int(data["step"])
    return result


async def counter_step(state: State, ctx: RunContext) -> Continue[State] | Complete[dict[str, int]]:
    await asyncio.sleep(0)
    if state.current >= state.target:
        return Complete({"current": state.current, "step": ctx.step_index})
    return Continue(State(current=state.current + 1, target=state.target))


def runner(
    tmp_path,
    step=counter_step,
    *,
    checkpoint_policy: CheckpointPolicy | None = None,
    payload_policy: PayloadPolicy | None = None,
) -> AgentRunner[State, dict[str, int]]:
    return AgentRunner(
        step=step,
        store=SQLiteCheckpointStore(tmp_path / "runs.sqlite"),
        encode_state=encode_state,
        decode_state=decode_state,
        encode_result=encode_result,
        decode_result=decode_result,
        checkpoint_policy=checkpoint_policy,
        payload_policy=payload_policy,
    )


def tool_runner(
    tmp_path,
    step,
    tools: ToolRegistry,
    *,
    payload_policy: PayloadPolicy | None = None,
) -> AgentRunner[State, dict[str, int]]:
    return AgentRunner(
        step=step,
        store=SQLiteCheckpointStore(tmp_path / "runs.sqlite"),
        encode_state=encode_state,
        decode_state=decode_state,
        encode_result=encode_result,
        decode_result=decode_result,
        tools=tools,
        payload_policy=payload_policy,
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
async def test_transient_step_failure_retries_and_commits_once(tmp_path) -> None:
    calls = 0

    async def flaky_step(state: State, ctx: RunContext) -> Continue[State]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StepExecutionError(
                "network wobble",
                error_category=ErrorCategory.TRANSIENT,
            )
        return Continue(State(current=state.current + 1, target=state.target))

    agent = runner(tmp_path, step=flaky_step)

    result = await agent.start(run_id="retry", initial_state=State(0, 10), max_steps=1)

    assert result.status == "paused"
    assert result.step_index == 1
    assert calls == 2
    attempts = agent.get_attempts("retry", step_index=0)
    assert [attempt.attempt_index for attempt in attempts] == [0, 1]
    assert [attempt.status for attempt in attempts] == ["failed", "completed"]
    assert attempts[0].error_category == "transient"
    assert len(agent.store.get_steps("retry")) == 2


@pytest.mark.asyncio
async def test_validation_error_does_not_retry(tmp_path) -> None:
    calls = 0

    async def validation_step(state: State, ctx: RunContext) -> Continue[State]:
        nonlocal calls
        calls += 1
        raise StepExecutionError(
            "bad shape",
            error_category=ErrorCategory.VALIDATION,
        )

    agent = runner(tmp_path, step=validation_step)

    with pytest.raises(StepExecutionError, match="bad shape"):
        await agent.start(run_id="validation", initial_state=State(0, 10), max_steps=1)

    assert calls == 1
    attempts = agent.get_attempts("validation", step_index=0)
    assert len(attempts) == 1
    assert attempts[0].error_category == "validation"
    assert agent.store.get_run("validation").status == "failed"


@pytest.mark.asyncio
async def test_max_attempts_exhausted_marks_run_failed(tmp_path) -> None:
    async def always_transient(state: State, ctx: RunContext) -> Continue[State]:
        raise StepExecutionError(
            "still down",
            error_category=ErrorCategory.TRANSIENT,
        )

    agent = runner(tmp_path, step=always_transient)

    with pytest.raises(StepExecutionError, match="still down"):
        await agent.start(run_id="exhausted", initial_state=State(0, 10), max_steps=1)

    attempts = agent.get_attempts("exhausted", step_index=0)
    assert [attempt.attempt_index for attempt in attempts] == [0, 1, 2]
    assert [attempt.status for attempt in attempts] == ["failed", "failed", "failed"]
    assert [attempt.is_retryable for attempt in attempts] == [True, True, True]
    stored = agent.store.get_run("exhausted")
    assert stored.status == "failed"
    assert stored.error == "StepExecutionError: still down"


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
async def test_runner_handles_100k_steps_without_recursion_failure(tmp_path) -> None:
    agent = runner(tmp_path)

    result = await agent.start(run_id="deep", initial_state=State(0, 100_000), max_steps=200_000)
    explanation = agent.explain_run("deep")

    assert result.status == "completed"
    assert result.step_index == 100_000
    assert explanation.checkpoint_count == 100_001
    assert explanation.attempt_count == 100_001
    assert explanation.invariant_notes == ()


@pytest.mark.asyncio
async def test_plain_async_recursion_fails_but_runtime_survives(tmp_path) -> None:
    async def plain_loop(state: State) -> State:
        if state.current >= state.target:
            return state
        return await plain_loop(State(current=state.current + 1, target=state.target))

    with pytest.raises(RecursionError):
        await plain_loop(State(0, sys.getrecursionlimit() + 100))

    agent = runner(tmp_path)
    result = await agent.start(
        run_id="stack-safe",
        initial_state=State(0, sys.getrecursionlimit() + 100),
        max_steps=sys.getrecursionlimit() + 200,
    )

    assert result.status == "completed"


@pytest.mark.asyncio
async def test_history_reconstructs_checkpoints_in_order(tmp_path) -> None:
    agent = runner(tmp_path)

    await agent.start(run_id="history", initial_state=State(0, 3), max_steps=10)

    history = agent.get_history("history")
    assert [step.step_index for step in history] == [0, 1, 2, 3]
    assert [step.state.current for step in history] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_history_pagination_returns_bounded_window(tmp_path) -> None:
    agent = runner(tmp_path)

    await agent.start(run_id="history-page", initial_state=State(0, 5), max_steps=10)

    history = agent.get_history("history-page", limit=2, offset=2)
    assert [step.step_index for step in history] == [2, 3]
    assert [step.state.current for step in history] == [2, 3]


@pytest.mark.asyncio
async def test_explain_run_returns_machine_readable_summary(tmp_path) -> None:
    agent = runner(tmp_path)

    await agent.start(run_id="explain", initial_state=State(0, 2), max_steps=10)

    explanation = agent.explain_run("explain")
    assert explanation.run_id == "explain"
    assert explanation.status == "completed"
    assert explanation.step_index == 2
    assert explanation.checkpoint_count == 3
    assert explanation.attempt_count == 3
    assert explanation.retry_count == 0
    assert explanation.tool_call_count == 0
    assert explanation.last_error_category is None
    assert explanation.invariant_notes == ()
    assert explanation.state is None
    assert explanation.result == {"current": 2, "step": 2}
    assert explanation.error is None


@pytest.mark.asyncio
async def test_get_stats_returns_aggregate_counts(tmp_path) -> None:
    agent = runner(tmp_path)

    await agent.start(run_id="stats", initial_state=State(0, 4), max_steps=10)

    stats = agent.get_stats("stats")
    assert stats.step_count == 5
    assert stats.attempt_count == 5
    assert stats.retry_count == 0
    assert stats.tool_call_count == 0
    assert stats.failed_attempt_count == 0
    assert stats.last_error_category is None


@pytest.mark.asyncio
async def test_explain_run_reports_retry_and_tool_counts(tmp_path) -> None:
    tools = ToolRegistry()
    tool_calls = 0
    step_calls = 0

    @tools.tool("value")
    def value() -> dict[str, int]:
        nonlocal tool_calls
        tool_calls += 1
        return {"current": 41}

    async def tool_then_retry(state: State, ctx: RunContext) -> Complete[dict[str, int]]:
        nonlocal step_calls
        step_calls += 1
        payload = await ctx.call_tool("value", idempotency_key="stable-tool")
        if step_calls == 1:
            raise StepExecutionError("retry me", error_category=ErrorCategory.TRANSIENT)
        return Complete({"current": payload["current"] + 1})

    agent = tool_runner(tmp_path, tool_then_retry, tools)

    result = await agent.start(run_id="explain-retry", initial_state=State(0, 1), max_steps=10)
    explanation = agent.explain_run("explain-retry")

    assert result.status == "completed"
    assert tool_calls == 1
    assert explanation.attempt_count == 2
    assert explanation.retry_count == 1
    assert explanation.tool_call_count == 1
    assert explanation.last_error_category == "transient"


@pytest.mark.asyncio
async def test_managed_tool_call_is_deduped_across_retry(tmp_path) -> None:
    tools = ToolRegistry()
    tool_calls = 0
    step_calls = 0

    @tools.tool("increment")
    def increment(value: int) -> dict[str, int]:
        nonlocal tool_calls
        tool_calls += 1
        return {"current": value + 1}

    async def tool_step(state: State, ctx: RunContext) -> Complete[dict[str, int]]:
        nonlocal step_calls
        step_calls += 1
        payload = await ctx.call_tool(
            "increment",
            state.current,
            idempotency_key="increment-once",
        )
        if step_calls == 1:
            raise StepExecutionError("after side effect", error_category=ErrorCategory.TRANSIENT)
        return Complete(payload)

    agent = tool_runner(tmp_path, tool_step, tools)

    result = await agent.start(run_id="tool-dedupe", initial_state=State(0, 1), max_steps=10)

    assert result.status == "completed"
    assert result.result == {"current": 1}
    assert step_calls == 2
    assert tool_calls == 1
    calls = agent.get_tool_calls("tool-dedupe", step_index=0)
    assert len(calls) == 1
    assert calls[0].idempotency_key == "increment-once"


@pytest.mark.asyncio
async def test_attempt_and_tool_call_pagination(tmp_path) -> None:
    tools = ToolRegistry()

    @tools.tool("echo")
    def echo(value: int) -> dict[str, int]:
        return {"value": value}

    async def paged_tool_step(state: State, ctx: RunContext) -> Continue[State] | Complete[dict[str, int]]:
        if state.current >= state.target:
            return Complete({"current": state.current})
        await ctx.call_tool("echo", state.current)
        return Continue(State(state.current + 1, state.target))

    agent = tool_runner(tmp_path, paged_tool_step, tools)

    await agent.start(run_id="page-runtime", initial_state=State(0, 4), max_steps=10)

    attempts = agent.get_attempts("page-runtime", limit=2, offset=1)
    tool_calls = agent.get_tool_calls("page-runtime", limit=2, offset=1)
    assert [attempt.step_index for attempt in attempts] == [1, 2]
    assert [call.step_index for call in tool_calls] == [1, 2]


@pytest.mark.asyncio
async def test_interval_checkpoint_policy_keeps_initial_interval_and_current_state(tmp_path) -> None:
    agent = runner(
        tmp_path,
        checkpoint_policy=CheckpointPolicy(mode="interval", every=3),
    )

    first = await agent.start(run_id="interval", initial_state=State(0, 10), max_steps=5)
    history = agent.get_history("interval")
    paused = agent.get_run("interval")
    resumed = await agent.resume(run_id="interval", max_steps=10)

    assert first.status == "paused"
    assert first.state == State(5, 10)
    assert [step.step_index for step in history] == [0, 3]
    assert paused.state == State(5, 10)
    assert resumed.status == "completed"
    assert resumed.result == {"current": 10, "step": 10}
    assert agent.verify_run("interval").valid is True


@pytest.mark.asyncio
async def test_payload_policy_truncates_large_tool_payload_and_dedupes_safe_shape(tmp_path) -> None:
    tools = ToolRegistry()
    tool_calls = 0

    @tools.tool("large")
    def large() -> dict[str, str]:
        nonlocal tool_calls
        tool_calls += 1
        return {"blob": "x" * 200}

    async def large_tool_step(state: State, ctx: RunContext) -> Complete[dict[str, Any]]:
        payload = await ctx.call_tool("large", idempotency_key="large-once")
        if ctx.attempt_index == 0:
            raise StepExecutionError("retry after large payload", error_category=ErrorCategory.TRANSIENT)
        return Complete(payload)

    agent = tool_runner(
        tmp_path,
        large_tool_step,
        tools,
        payload_policy=PayloadPolicy(max_inline_bytes=32),
    )

    result = await agent.start(run_id="large-payload", initial_state=State(0, 1), max_steps=10)
    calls = agent.get_tool_calls("large-payload")

    assert result.status == "completed"
    assert tool_calls == 1
    assert result.result is not None
    assert result.result["payload_truncated"] is True
    assert result.result["payload_bytes"] > 32
    assert len(calls) == 1
    assert '"blob"' not in (calls[0].result_json or "")


@pytest.mark.asyncio
async def test_managed_tool_error_envelope_drives_retry_taxonomy(tmp_path) -> None:
    tools = ToolRegistry()
    tool_calls = 0

    @tools.tool("unstable")
    def unstable() -> dict:
        nonlocal tool_calls
        tool_calls += 1
        if tool_calls == 1:
            return {
                "success": False,
                "is_error": True,
                "error_category": "transient",
                "is_retryable": True,
                "payload": {"reason": "try again"},
            }
        return {"success": True, "payload": {"current": 7}}

    async def tool_step(state: State, ctx: RunContext) -> Complete[dict[str, int]]:
        payload = await ctx.call_tool("unstable", idempotency_key=f"unstable-{ctx.attempt_index}")
        return Complete(payload)

    agent = tool_runner(tmp_path, tool_step, tools)

    result = await agent.start(run_id="tool-taxonomy", initial_state=State(0, 1), max_steps=10)

    assert result.status == "completed"
    assert result.result == {"current": 7}
    assert tool_calls == 2
    attempts = agent.get_attempts("tool-taxonomy", step_index=0)
    assert [attempt.status for attempt in attempts] == ["failed", "completed"]
    assert attempts[0].error_category == "transient"


@pytest.mark.asyncio
async def test_verify_run_reports_clean_runtime_invariants(tmp_path) -> None:
    agent = runner(tmp_path)

    await agent.start(run_id="verify-clean", initial_state=State(0, 2), max_steps=10)

    verification = agent.verify_run("verify-clean")
    assert verification.valid is True
    assert verification.notes == ()


@pytest.mark.asyncio
async def test_get_and_list_runs_decode_current_runtime_state(tmp_path) -> None:
    agent = runner(tmp_path)

    await agent.start(run_id="paused", initial_state=State(0, 10), max_steps=2)
    await agent.start(run_id="done", initial_state=State(0, 1), max_steps=10)

    paused = agent.get_run("paused")
    all_runs = agent.list_runs()

    assert paused.status == "paused"
    assert paused.state == State(2, 10)
    assert [run.run_id for run in all_runs] == ["paused", "done"]
    assert [run.status for run in all_runs] == ["paused", "completed"]


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
