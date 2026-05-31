from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from flow_xray import trace
from loom import tailrec

from .checkpoint import (
    CheckpointStore,
    SQLiteCheckpointStore,
    StoredAttempt,
    StoredCommittedStep,
    StoredRun,
    StoredStep,
    StoredToolCall,
)
from .errors import (
    ErrorCategory,
    RetryPolicy,
    StepExecutionError,
    error_category_for,
    is_retryable_error,
)
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


@dataclass
class RunContext:
    run_id: str
    step_index: int
    tools: ToolRegistry
    payload_policy: PayloadPolicy | None = None
    attempt_index: int = 0
    input_hash: str = ""
    idempotency_key: str = ""
    store: CheckpointStore | None = None
    _tool_call_index: int = field(default=0, init=False, repr=False)

    async def call_tool(
        self,
        name: str,
        *args: Any,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.store is None:
            return await self.tools.call(name, *args, **kwargs)

        call_index = self._tool_call_index
        self._tool_call_index += 1
        request_json = _stable_json({"args": args, "kwargs": kwargs})
        request_hash = _hash_text(request_json)
        key = idempotency_key or (
            f"{self.run_id}:{self.step_index}:tool:{call_index}:{name}:{request_hash}"
        )
        stored = self.store.get_tool_call_by_key(key)
        if stored is not None and stored.result_json is not None:
            return json.loads(stored.result_json)["payload"]

        tool_call_id = f"{key}:attempt:{self.attempt_index}:call:{call_index}"
        self.store.record_tool_call_started(
            tool_call_id=tool_call_id,
            run_id=self.run_id,
            step_index=self.step_index,
            attempt_index=self.attempt_index,
            idempotency_key=key,
            tool_name=name,
            request_hash=request_hash,
        )
        try:
            result = await self.tools.call_result(name, *args, **kwargs)
        except Exception as exc:
            category = error_category_for(exc)
            self.store.record_tool_call_finished(
                tool_call_id=tool_call_id,
                status="failed",
                result_json=None,
                error=f"{exc.__class__.__name__}: {exc}",
                error_category=category.value,
            )
            raise

        if result.is_error:
            category = result.error_category or ErrorCategory.UNKNOWN.value
            self.store.record_tool_call_finished(
                tool_call_id=tool_call_id,
                status="error",
                result_json=_stable_json(_apply_payload_policy(result.to_dict(), self.payload_policy)),
                error=f"tool returned error: {name}",
                error_category=category,
            )
            raise StepExecutionError(
                f"tool returned error: {name}",
                error_category=category,
                is_retryable=result.is_retryable,
            )

        stored_result = _apply_payload_policy(result.to_dict(), self.payload_policy)
        self.store.record_tool_call_finished(
            tool_call_id=tool_call_id,
            status="completed",
            result_json=_stable_json(stored_result),
            error=None,
            error_category=None,
        )
        return stored_result["payload"]


@dataclass(frozen=True)
class RunResult(Generic[S, R]):
    run_id: str
    status: Literal["completed", "paused", "failed"]
    step_index: int
    state: S | None
    result: R | None


@dataclass(frozen=True)
class RunStep(Generic[S]):
    run_id: str
    step_index: int
    state: S


@dataclass(frozen=True)
class RunExplanation(Generic[S, R]):
    run_id: str
    status: str
    step_index: int
    checkpoint_count: int
    attempt_count: int
    retry_count: int
    tool_call_count: int
    last_error_category: str | None
    invariant_notes: tuple[str, ...]
    state: S | None
    result: R | None
    error: str | None


@dataclass(frozen=True)
class RunStats:
    step_count: int
    attempt_count: int
    retry_count: int
    tool_call_count: int
    failed_attempt_count: int
    last_error_category: str | None


@dataclass(frozen=True)
class RunVerification:
    run_id: str
    valid: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointPolicy:
    mode: Literal["full", "interval", "compact"] = "full"
    every: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"full", "interval", "compact"}:
            raise ValueError(
                "checkpoint policy mode must be 'full', 'interval', or 'compact'"
            )
        if self.every < 1:
            raise ValueError("checkpoint policy every must be >= 1")

    def should_retain(self, step_index: int) -> bool:
        if self.mode == "interval":
            return step_index == 0 or step_index % self.every == 0
        return True

    def should_persist_attempt(self, *, attempt_index: int, status: str) -> bool:
        if self.mode != "compact":
            return True
        if status == "failed":
            return True
        if attempt_index > 0:
            return True
        return False

    def should_persist_attempt_started(self, *, attempt_index: int) -> bool:
        if self.mode != "compact":
            return True
        return attempt_index > 0


@dataclass(frozen=True)
class PayloadPolicy:
    max_inline_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_inline_bytes is not None and self.max_inline_bytes < 0:
            raise ValueError("max_inline_bytes must be >= 0")


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
        store: CheckpointStore,
        encode_state: Encoder,
        decode_state: Decoder,
        encode_result: Encoder,
        decode_result: Decoder,
        tools: ToolRegistry | None = None,
        retry_policy: RetryPolicy | None = None,
        checkpoint_policy: CheckpointPolicy | None = None,
        payload_policy: PayloadPolicy | None = None,
    ) -> None:
        self.step = step
        self.store = store
        self.encode_state = encode_state
        self.decode_state = decode_state
        self.encode_result = encode_result
        self.decode_result = decode_result
        self.tools = tools or ToolRegistry()
        self.retry_policy = retry_policy or RetryPolicy()
        self.checkpoint_policy = checkpoint_policy or CheckpointPolicy()
        self.payload_policy = payload_policy or PayloadPolicy()

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

    def get_run(self, run_id: str) -> RunResult[S, R]:
        return self._decode_stored_run(self.store.get_run(run_id))

    def list_runs(self) -> list[RunResult[S, R]]:
        return [self._decode_stored_run(stored) for stored in self.store.list_runs()]

    def get_history(
        self, run_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[RunStep[S]]:
        return [
            self._decode_stored_step(step)
            for step in self.store.get_steps(run_id, limit=limit, offset=offset)
        ]

    def get_attempts(
        self,
        run_id: str,
        step_index: int | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StoredAttempt]:
        return self.store.get_attempts(run_id, step_index, limit=limit, offset=offset)

    def get_tool_calls(
        self,
        run_id: str,
        step_index: int | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StoredToolCall]:
        return self.store.get_tool_calls(run_id, step_index, limit=limit, offset=offset)

    def get_stats(self, run_id: str) -> RunStats:
        last_failed = self.store.last_failed_attempt(run_id)
        return RunStats(
            step_count=self.store.count_steps(run_id),
            attempt_count=self.store.count_attempts(run_id),
            retry_count=self.store.count_retries(run_id),
            tool_call_count=self.store.count_tool_calls(run_id),
            failed_attempt_count=self.store.count_failed_attempts(run_id),
            last_error_category=None if last_failed is None else last_failed.error_category,
        )

    def verify_run(self, run_id: str) -> RunVerification:
        stored = self.store.get_run(run_id)
        notes: list[str] = []

        if self.checkpoint_policy.mode in {"full", "compact"} and self.store.has_step_gaps(
            run_id
        ):
            notes.append("step_gap")

        if stored.status == "completed":
            if stored.state_json is not None:
                notes.append("completed_run_has_state")
            if stored.result_json is None:
                notes.append("completed_run_missing_result")
        elif stored.status in {"running", "paused"}:
            if stored.state_json is None:
                notes.append("resumable_run_missing_state")
        elif stored.status == "failed":
            if stored.error is None:
                notes.append("failed_run_missing_error")
        else:
            notes.append("unknown_run_status")

        if self.checkpoint_policy.mode != "compact" and self.store.has_attempt_gaps(
            run_id
        ):
            notes.append("attempt_gap")

        return RunVerification(run_id=run_id, valid=not notes, notes=tuple(notes))

    def explain_run(self, run_id: str) -> RunExplanation[S, R]:
        stored = self.store.get_run(run_id)
        decoded = self._decode_stored_run(stored)
        stats = self.get_stats(run_id)
        verification = self.verify_run(run_id)
        notes = list(verification.notes)
        if self.checkpoint_policy.mode == "compact":
            notes.append("compact_attempt_log")
        return RunExplanation(
            run_id=stored.run_id,
            status=stored.status,
            step_index=stored.step_index,
            checkpoint_count=stats.step_count,
            attempt_count=stats.attempt_count,
            retry_count=stats.retry_count,
            tool_call_count=stats.tool_call_count,
            last_error_category=stats.last_error_category,
            invariant_notes=tuple(notes),
            state=decoded.state,
            result=decoded.result,
            error=stored.error,
        )

    @trace
    async def _step_once(
        self, cursor: _Cursor[S, R], *, attempt_index: int, input_hash: str
    ) -> Continue[S] | Complete[R]:
        ctx = RunContext(
            run_id=cursor.run_id,
            step_index=cursor.step_index,
            tools=self.tools,
            attempt_index=attempt_index,
            input_hash=input_hash,
            idempotency_key=f"{cursor.run_id}:{cursor.step_index}:{input_hash}",
            store=self.store,
            payload_policy=self.payload_policy,
        )
        trace.meta(
            run_id=cursor.run_id,
            step_index=cursor.step_index,
            attempt_index=attempt_index,
            input_hash=input_hash,
        )
        return await self.step(cursor.state, ctx)

    async def _step_with_attempts(self, cursor: _Cursor[S, R]) -> Continue[S] | Complete[R]:
        state_json = self._dumps_state(cursor.state)
        input_hash = _hash_text(
            _stable_json(
                {
                    "runtime": "loom-agent-runtime-v1",
                    "state": json.loads(state_json),
                }
            )
        )
        committed = self.store.get_committed_step(
            run_id=cursor.run_id,
            step_index=cursor.step_index,
            input_hash=input_hash,
        )
        if committed is not None:
            return self._decode_committed_outcome(committed)

        attempt_index = 0
        while True:
            if self.checkpoint_policy.should_persist_attempt_started(
                attempt_index=attempt_index
            ):
                self.store.record_attempt_started(
                    run_id=cursor.run_id,
                    step_index=cursor.step_index,
                    attempt_index=attempt_index,
                    input_hash=input_hash,
                )
            try:
                outcome = await self._step_once(
                    cursor,
                    attempt_index=attempt_index,
                    input_hash=input_hash,
                )
                if not isinstance(outcome, (Continue, Complete)):
                    raise TypeError(
                        f"step must return Continue or Complete, got {type(outcome).__name__}"
                    )

                outcome_kind, state_json, result_json = self._encode_outcome(outcome)
                if self.checkpoint_policy.should_persist_attempt(
                    attempt_index=attempt_index,
                    status="completed",
                ):
                    self.store.record_attempt_finished(
                        run_id=cursor.run_id,
                        step_index=cursor.step_index,
                        attempt_index=attempt_index,
                        status="completed",
                        outcome_kind=outcome_kind,
                        output_json=state_json if outcome_kind == "continue" else result_json,
                        error=None,
                        error_category=None,
                        is_retryable=None,
                    )
                self.store.commit_step_outcome(
                    run_id=cursor.run_id,
                    step_index=cursor.step_index,
                    input_hash=input_hash,
                    outcome_kind=outcome_kind,
                    state_json=state_json,
                    result_json=result_json,
                    retain_checkpoint=self._should_retain_checkpoint(cursor, outcome),
                )
                return outcome
            except Exception as exc:
                category = error_category_for(exc)
                error_is_retryable = is_retryable_error(exc)
                retryable = error_is_retryable and self.retry_policy.should_retry(
                    category, attempt_index
                )
                if self.checkpoint_policy.should_persist_attempt(
                    attempt_index=attempt_index,
                    status="failed",
                ):
                    if not self.checkpoint_policy.should_persist_attempt_started(
                        attempt_index=attempt_index
                    ):
                        self.store.record_attempt_started(
                            run_id=cursor.run_id,
                            step_index=cursor.step_index,
                            attempt_index=attempt_index,
                            input_hash=input_hash,
                        )
                    self.store.record_attempt_finished(
                        run_id=cursor.run_id,
                        step_index=cursor.step_index,
                        attempt_index=attempt_index,
                        status="failed",
                        outcome_kind=None,
                        output_json=None,
                        error=f"{exc.__class__.__name__}: {exc}",
                        error_category=category.value,
                        is_retryable=error_is_retryable,
                    )
                if retryable:
                    attempt_index += 1
                    continue
                raise

    def _save_continue(self, cursor: _Cursor[S, R], outcome: Continue[S]) -> _Cursor[S, R]:
        next_step = cursor.step_index + 1
        return _Cursor(
            run_id=cursor.run_id,
            state=outcome.state,
            step_index=next_step,
            remaining_steps=cursor.remaining_steps - 1,
        )

    def _complete(self, cursor: _Cursor[S, R], outcome: Complete[R]) -> RunResult[S, R]:
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

    def _decode_stored_run(self, stored: StoredRun) -> RunResult[S, R]:
        state = None
        result = None
        if stored.state_json is not None:
            state = self.decode_state(json.loads(stored.state_json))
        if stored.result_json is not None:
            result = self.decode_result(json.loads(stored.result_json))
        return RunResult(
            run_id=stored.run_id,
            status=stored.status,  # type: ignore[arg-type]
            step_index=stored.step_index,
            state=state,
            result=result,
        )

    def _decode_stored_step(self, step: StoredStep) -> RunStep[S]:
        return RunStep(
            run_id=step.run_id,
            step_index=step.step_index,
            state=self.decode_state(json.loads(step.state_json)),
        )

    def _decode_committed_outcome(
        self, committed: StoredCommittedStep
    ) -> Continue[S] | Complete[R]:
        if committed.outcome_kind == "continue":
            if committed.state_json is None:
                raise ValueError("committed continue outcome has no state")
            return Continue(self.decode_state(json.loads(committed.state_json)))
        if committed.outcome_kind == "complete":
            if committed.result_json is None:
                raise ValueError("committed complete outcome has no result")
            return Complete(self.decode_result(json.loads(committed.result_json)))
        raise ValueError(f"unknown committed outcome kind: {committed.outcome_kind}")

    def _encode_outcome(
        self, outcome: Continue[S] | Complete[R]
    ) -> tuple[Literal["continue", "complete"], str | None, str | None]:
        if isinstance(outcome, Continue):
            return "continue", self._dumps_state(outcome.state), None
        return "complete", None, _stable_json(self.encode_result(outcome.result))

    def _should_retain_checkpoint(
        self, cursor: _Cursor[S, R], outcome: Continue[S] | Complete[R]
    ) -> bool:
        if isinstance(outcome, Complete):
            return False
        return self.checkpoint_policy.should_retain(cursor.step_index + 1)


@tailrec
async def _drive(cursor: _Cursor[S, R], runner: AgentRunner[S, R]) -> RunResult[S, R]:
    if cursor.remaining_steps <= 0:
        return runner._pause(cursor)

    try:
        outcome = await runner._step_with_attempts(cursor)
        if isinstance(outcome, Complete):
            return runner._complete(cursor, outcome)
        next_cursor = runner._save_continue(cursor, outcome)
    except Exception as exc:
        runner._fail(cursor, exc)
        raise

    return await _drive(next_cursor, runner)


def _validate_max_steps(max_steps: int) -> None:
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _apply_payload_policy(
    result: dict[str, Any], policy: PayloadPolicy | None
) -> dict[str, Any]:
    if policy is None or policy.max_inline_bytes is None:
        return result

    payload_json = _stable_json(result.get("payload"))
    payload_bytes = payload_json.encode("utf-8")
    if len(payload_bytes) <= policy.max_inline_bytes:
        return result

    updated = dict(result)
    metadata = dict(updated.get("metadata", {}))
    metadata.update(
        {
            "payload_truncated": True,
            "payload_hash": _hash_text(payload_json),
            "payload_bytes": len(payload_bytes),
        }
    )
    updated["metadata"] = metadata
    updated["payload"] = {
        "payload_truncated": True,
        "payload_hash": metadata["payload_hash"],
        "payload_bytes": metadata["payload_bytes"],
    }
    return updated
