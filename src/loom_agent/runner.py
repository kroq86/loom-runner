from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from flow_xray import trace
from loom import tailrec

from .checkpoint import SQLiteCheckpointStore
from .tools import ToolRegistry


S = TypeVar("S")
R = TypeVar("R")
JsonValue: TypeAlias = dict[str, Any] | list[Any] | str | int | float | bool | None


@dataclass(frozen=True)
class Continue(Generic[S]):
    state: S


@dataclass(frozen=True)
class Complete(Generic[R]):
    result: R


@dataclass(frozen=True)
class RunContext:
    run_id: str
    step_index: int
    tools: ToolRegistry


@dataclass(frozen=True)
class RunResult(Generic[S, R]):
    run_id: str
    status: Literal["completed", "paused", "failed"]
    step_index: int
    state: S | None
    result: R | None


@dataclass(frozen=True)
class _Cursor(Generic[S, R]):
    run_id: str
    state: S
    step_index: int
    remaining_steps: int


StepFn: TypeAlias = Callable[[S, RunContext], Awaitable[Continue[S] | Complete[R]]]
Encoder: TypeAlias = Callable[[Any], JsonValue]
Decoder: TypeAlias = Callable[[JsonValue], Any]


class AgentRunner(Generic[S, R]):
    def __init__(
        self,
        *,
        step: StepFn[S, R],
        store: SQLiteCheckpointStore,
        encode_state: Encoder,
        decode_state: Decoder,
        encode_result: Encoder,
        decode_result: Decoder,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.step = step
        self.store = store
        self.encode_state = encode_state
        self.decode_state = decode_state
        self.encode_result = encode_result
        self.decode_result = decode_result
        self.tools = tools or ToolRegistry()

    async def start(self, *, run_id: str, initial_state: S, max_steps: int) -> RunResult[S, R]:
        _validate_max_steps(max_steps)
        self.store.create_run(run_id=run_id, state_json=self._dumps_state(initial_state))
        cursor: _Cursor[S, R] = _Cursor(
            run_id=run_id,
            state=initial_state,
            step_index=0,
            remaining_steps=max_steps,
        )
        return await _drive(cursor, self)

    async def resume(self, *, run_id: str, max_steps: int) -> RunResult[S, R]:
        _validate_max_steps(max_steps)
        stored = self.store.get_run(run_id)
        if stored.status == "completed":
            raise ValueError(f"run already completed: {run_id}")
        if stored.status == "failed":
            raise ValueError(f"run is failed and cannot be resumed: {run_id}")
        if stored.state_json is None:
            raise ValueError(f"run has no resumable state: {run_id}")

        cursor: _Cursor[S, R] = _Cursor(
            run_id=run_id,
            state=self.decode_state(json.loads(stored.state_json)),
            step_index=stored.step_index,
            remaining_steps=max_steps,
        )
        return await _drive(cursor, self)

    @trace
    async def _step_once(self, cursor: _Cursor[S, R]) -> Continue[S] | Complete[R]:
        ctx = RunContext(
            run_id=cursor.run_id,
            step_index=cursor.step_index,
            tools=self.tools,
        )
        trace.meta(run_id=cursor.run_id, step_index=cursor.step_index)
        return await self.step(cursor.state, ctx)

    def _save_continue(self, cursor: _Cursor[S, R], outcome: Continue[S]) -> _Cursor[S, R]:
        next_step = cursor.step_index + 1
        self.store.save_step(
            run_id=cursor.run_id,
            step_index=next_step,
            state_json=self._dumps_state(outcome.state),
        )
        return _Cursor(
            run_id=cursor.run_id,
            state=outcome.state,
            step_index=next_step,
            remaining_steps=cursor.remaining_steps - 1,
        )

    def _complete(self, cursor: _Cursor[S, R], outcome: Complete[R]) -> RunResult[S, R]:
        self.store.mark_completed(
            run_id=cursor.run_id,
            step_index=cursor.step_index,
            result_json=json.dumps(self.encode_result(outcome.result), sort_keys=True),
        )
        return RunResult(
            run_id=cursor.run_id,
            status="completed",
            step_index=cursor.step_index,
            state=None,
            result=outcome.result,
        )

    def _pause(self, cursor: _Cursor[S, R]) -> RunResult[S, R]:
        self.store.mark_paused(run_id=cursor.run_id)
        return RunResult(
            run_id=cursor.run_id,
            status="paused",
            step_index=cursor.step_index,
            state=cursor.state,
            result=None,
        )

    def _fail(self, cursor: _Cursor[S, R], exc: BaseException) -> None:
        self.store.mark_failed(run_id=cursor.run_id, error=f"{exc.__class__.__name__}: {exc}")

    def _dumps_state(self, state: S) -> str:
        return json.dumps(self.encode_state(state), sort_keys=True)


@tailrec
async def _drive(cursor: _Cursor[S, R], runner: AgentRunner[S, R]) -> RunResult[S, R]:
    if cursor.remaining_steps <= 0:
        return runner._pause(cursor)

    try:
        outcome = await runner._step_once(cursor)
        if isinstance(outcome, Complete):
            return runner._complete(cursor, outcome)
        if not isinstance(outcome, Continue):
            raise TypeError(f"step must return Continue or Complete, got {type(outcome).__name__}")
        next_cursor = runner._save_continue(cursor, outcome)
    except Exception as exc:
        runner._fail(cursor, exc)
        raise

    return await _drive(next_cursor, runner)


def _validate_max_steps(max_steps: int) -> None:
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")
