from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    status: str
    step_index: int
    state_json: str | None
    result_json: str | None
    error: str | None


@dataclass(frozen=True)
class StoredStep:
    run_id: str
    step_index: int
    state_json: str
    created_at: str


@dataclass(frozen=True)
class StoredCommittedStep:
    run_id: str
    step_index: int
    input_hash: str
    outcome_kind: str
    state_json: str | None
    result_json: str | None
    created_at: str


@dataclass(frozen=True)
class StoredAttempt:
    run_id: str
    step_index: int
    attempt_index: int
    input_hash: str
    status: str
    outcome_kind: str | None
    output_json: str | None
    error: str | None
    error_category: str | None
    is_retryable: bool | None
    started_at: str
    finished_at: str | None


@dataclass(frozen=True)
class StoredToolCall:
    tool_call_id: str
    run_id: str
    step_index: int
    attempt_index: int
    idempotency_key: str
    tool_name: str
    request_hash: str
    status: str
    result_json: str | None
    error: str | None
    error_category: str | None
    started_at: str
    finished_at: str | None


class CheckpointStore(Protocol):
    """Persistence contract required by the Loom agent runtime."""

    def create_run(self, *, run_id: str, state_json: str) -> None: ...

    def get_run(self, run_id: str) -> StoredRun: ...

    def list_runs(self) -> list[StoredRun]: ...

    def get_steps(
        self, run_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[StoredStep]: ...

    def get_attempts(
        self,
        run_id: str,
        step_index: int | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StoredAttempt]: ...

    def get_tool_calls(
        self,
        run_id: str,
        step_index: int | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StoredToolCall]: ...

    def get_tool_call_by_key(self, idempotency_key: str) -> StoredToolCall | None: ...

    def get_committed_step(
        self, *, run_id: str, step_index: int, input_hash: str
    ) -> StoredCommittedStep | None: ...

    def save_step(self, *, run_id: str, step_index: int, state_json: str) -> None: ...

    def record_attempt_started(
        self, *, run_id: str, step_index: int, attempt_index: int, input_hash: str
    ) -> None: ...

    def record_attempt_finished(
        self,
        *,
        run_id: str,
        step_index: int,
        attempt_index: int,
        status: str,
        outcome_kind: str | None,
        output_json: str | None,
        error: str | None,
        error_category: str | None,
        is_retryable: bool | None,
    ) -> None: ...

    def commit_step_outcome(
        self,
        *,
        run_id: str,
        step_index: int,
        input_hash: str,
        outcome_kind: str,
        state_json: str | None,
        result_json: str | None,
        retain_checkpoint: bool = True,
    ) -> None: ...

    def record_tool_call_started(
        self,
        *,
        tool_call_id: str,
        run_id: str,
        step_index: int,
        attempt_index: int,
        idempotency_key: str,
        tool_name: str,
        request_hash: str,
    ) -> None: ...

    def record_tool_call_finished(
        self,
        *,
        tool_call_id: str,
        status: str,
        result_json: str | None,
        error: str | None,
        error_category: str | None,
    ) -> None: ...

    def mark_paused(self, *, run_id: str) -> None: ...

    def mark_completed(self, *, run_id: str, step_index: int, result_json: str) -> None: ...

    def mark_failed(self, *, run_id: str, error: str) -> None: ...

    def count_steps(self, run_id: str) -> int: ...

    def count_attempts(self, run_id: str) -> int: ...

    def count_tool_calls(self, run_id: str) -> int: ...

    def count_retries(self, run_id: str) -> int: ...

    def count_failed_attempts(self, run_id: str) -> int: ...

    def last_failed_attempt(self, run_id: str) -> StoredAttempt | None: ...

    def has_step_gaps(self, run_id: str) -> bool: ...

    def has_attempt_gaps(self, run_id: str) -> bool: ...


_SCHEMA_VERSION = 2


class SQLiteCheckpointStore:
    """SQLite-backed checkpoint store for JSON-encoded run state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def create_run(self, *, run_id: str, state_json: str) -> None:
        now = _now()
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    insert into runs(
                        run_id, status, step_index, state_json, result_json,
                        error, created_at, updated_at
                    )
                    values (?, 'running', 0, ?, null, null, ?, ?)
                    """,
                    (run_id, state_json, now, now),
                )
                conn.execute(
                    """
                    insert into steps(run_id, step_index, state_json, created_at)
                    values (?, 0, ?, ?)
                    """,
                    (run_id, state_json, now),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"run already exists: {run_id}") from exc

    def get_run(self, run_id: str) -> StoredRun:
        with self._connect() as conn:
            row = conn.execute(
                """
                select run_id, status, step_index, state_json, result_json, error
                from runs
                where run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"run not found: {run_id}")
        return StoredRun(
            run_id=str(row["run_id"]),
            status=str(row["status"]),
            step_index=int(row["step_index"]),
            state_json=row["state_json"],
            result_json=row["result_json"],
            error=row["error"],
        )

    def list_runs(self) -> list[StoredRun]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select run_id, status, step_index, state_json, result_json, error
                from runs
                order by created_at, run_id
                """
            ).fetchall()
        return [
            StoredRun(
                run_id=str(row["run_id"]),
                status=str(row["status"]),
                step_index=int(row["step_index"]),
                state_json=row["state_json"],
                result_json=row["result_json"],
                error=row["error"],
            )
            for row in rows
        ]

    def get_steps(
        self, run_id: str, *, limit: int | None = None, offset: int = 0
    ) -> list[StoredStep]:
        self.get_run(run_id)
        clause, extra_params = _limit_clause(limit=limit, offset=offset)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select run_id, step_index, state_json, created_at
                from steps
                where run_id = ?
                order by step_index
                {clause}
                """,
                (run_id, *extra_params),
            ).fetchall()
        return [
            StoredStep(
                run_id=str(row["run_id"]),
                step_index=int(row["step_index"]),
                state_json=str(row["state_json"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def get_attempts(
        self,
        run_id: str,
        step_index: int | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StoredAttempt]:
        self.get_run(run_id)
        query = """
            select run_id, step_index, attempt_index, input_hash, status,
                   outcome_kind, output_json, error, error_category,
                   is_retryable, started_at, finished_at
            from attempts
            where run_id = ?
        """
        params: tuple[object, ...] = (run_id,)
        if step_index is not None:
            query += " and step_index = ?"
            params = (run_id, step_index)
        query += " order by step_index, attempt_index"
        clause, extra_params = _limit_clause(limit=limit, offset=offset)
        query += clause
        with self._connect() as conn:
            rows = conn.execute(query, (*params, *extra_params)).fetchall()
        return [
            self._stored_attempt(row)
            for row in rows
        ]

    def get_tool_calls(
        self,
        run_id: str,
        step_index: int | None = None,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StoredToolCall]:
        self.get_run(run_id)
        query = """
            select tool_call_id, run_id, step_index, attempt_index,
                   idempotency_key, tool_name, request_hash, status,
                   result_json, error, error_category, started_at, finished_at
            from tool_calls
            where run_id = ?
        """
        params: tuple[object, ...] = (run_id,)
        if step_index is not None:
            query += " and step_index = ?"
            params = (run_id, step_index)
        query += " order by step_index, attempt_index, started_at, tool_call_id"
        clause, extra_params = _limit_clause(limit=limit, offset=offset)
        query += clause
        with self._connect() as conn:
            rows = conn.execute(query, (*params, *extra_params)).fetchall()
        return [self._stored_tool_call(row) for row in rows]

    def get_tool_call_by_key(self, idempotency_key: str) -> StoredToolCall | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                select tool_call_id, run_id, step_index, attempt_index,
                       idempotency_key, tool_name, request_hash, status,
                       result_json, error, error_category, started_at, finished_at
                from tool_calls
                where idempotency_key = ? and status = 'completed'
                order by finished_at desc, tool_call_id desc
                limit 1
                """,
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return self._stored_tool_call(row)

    def get_committed_step(
        self, *, run_id: str, step_index: int, input_hash: str
    ) -> StoredCommittedStep | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                select run_id, step_index, input_hash, outcome_kind,
                       state_json, result_json, created_at
                from committed_steps
                where run_id = ? and step_index = ? and input_hash = ?
                """,
                (run_id, step_index, input_hash),
            ).fetchone()
        if row is None:
            return None
        return StoredCommittedStep(
            run_id=str(row["run_id"]),
            step_index=int(row["step_index"]),
            input_hash=str(row["input_hash"]),
            outcome_kind=str(row["outcome_kind"]),
            state_json=row["state_json"],
            result_json=row["result_json"],
            created_at=str(row["created_at"]),
        )

    def save_step(self, *, run_id: str, step_index: int, state_json: str) -> None:
        now = _now()
        with self._connect() as conn:
            self._save_step_on_connection(
                conn=conn,
                run_id=run_id,
                step_index=step_index,
                state_json=state_json,
                now=now,
            )

    def record_attempt_started(
        self, *, run_id: str, step_index: int, attempt_index: int, input_hash: str
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into attempts(
                    run_id, step_index, attempt_index, input_hash, status,
                    outcome_kind, output_json, error, error_category,
                    is_retryable, started_at, finished_at
                )
                values (?, ?, ?, ?, 'started', null, null, null, null, null, ?, null)
                """,
                (run_id, step_index, attempt_index, input_hash, now),
            )

    def record_attempt_finished(
        self,
        *,
        run_id: str,
        step_index: int,
        attempt_index: int,
        status: str,
        outcome_kind: str | None,
        output_json: str | None,
        error: str | None,
        error_category: str | None,
        is_retryable: bool | None,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            updated = conn.execute(
                """
                update attempts
                set status = ?,
                    outcome_kind = ?,
                    output_json = ?,
                    error = ?,
                    error_category = ?,
                    is_retryable = ?,
                    finished_at = ?
                where run_id = ? and step_index = ? and attempt_index = ?
                """,
                (
                    status,
                    outcome_kind,
                    output_json,
                    error,
                    error_category,
                    None if is_retryable is None else int(is_retryable),
                    now,
                    run_id,
                    step_index,
                    attempt_index,
                ),
            )
            if updated.rowcount == 0:
                raise KeyError(
                    f"attempt not found: {run_id}:{step_index}:{attempt_index}"
                )

    def commit_step_outcome(
        self,
        *,
        run_id: str,
        step_index: int,
        input_hash: str,
        outcome_kind: str,
        state_json: str | None,
        result_json: str | None,
        retain_checkpoint: bool = True,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into committed_steps(
                    run_id, step_index, input_hash, outcome_kind,
                    state_json, result_json, created_at
                )
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, step_index, input_hash, outcome_kind, state_json, result_json, now),
            )
            if outcome_kind == "continue":
                required_state_json = _require_json(state_json, "state_json")
                if retain_checkpoint:
                    self._save_step_on_connection(
                        conn=conn,
                        run_id=run_id,
                        step_index=step_index + 1,
                        state_json=required_state_json,
                        now=now,
                    )
                else:
                    self._update_run_state_on_connection(
                        conn=conn,
                        run_id=run_id,
                        step_index=step_index + 1,
                        state_json=required_state_json,
                        now=now,
                    )
            elif outcome_kind == "complete":
                self._mark_completed_on_connection(
                    conn=conn,
                    run_id=run_id,
                    step_index=step_index,
                    result_json=_require_json(result_json, "result_json"),
                    now=now,
                )
            else:
                raise ValueError(f"unknown outcome kind: {outcome_kind}")

    def record_tool_call_started(
        self,
        *,
        tool_call_id: str,
        run_id: str,
        step_index: int,
        attempt_index: int,
        idempotency_key: str,
        tool_name: str,
        request_hash: str,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into tool_calls(
                    tool_call_id, run_id, step_index, attempt_index,
                    idempotency_key, tool_name, request_hash, status,
                    result_json, error, error_category, started_at, finished_at
                )
                values (?, ?, ?, ?, ?, ?, ?, 'started', null, null, null, ?, null)
                """,
                (
                    tool_call_id,
                    run_id,
                    step_index,
                    attempt_index,
                    idempotency_key,
                    tool_name,
                    request_hash,
                    now,
                ),
            )

    def record_tool_call_finished(
        self,
        *,
        tool_call_id: str,
        status: str,
        result_json: str | None,
        error: str | None,
        error_category: str | None,
    ) -> None:
        now = _now()
        with self._connect() as conn:
            updated = conn.execute(
                """
                update tool_calls
                set status = ?,
                    result_json = ?,
                    error = ?,
                    error_category = ?,
                    finished_at = ?
                where tool_call_id = ?
                """,
                (status, result_json, error, error_category, now, tool_call_id),
            )
            if updated.rowcount == 0:
                raise KeyError(f"tool call not found: {tool_call_id}")

    def mark_paused(self, *, run_id: str) -> None:
        self._update_status(run_id=run_id, status="paused")

    def mark_completed(self, *, run_id: str, step_index: int, result_json: str) -> None:
        now = _now()
        with self._connect() as conn:
            self._mark_completed_on_connection(
                conn=conn,
                run_id=run_id,
                step_index=step_index,
                result_json=result_json,
                now=now,
            )

    def mark_failed(self, *, run_id: str, error: str) -> None:
        now = _now()
        with self._connect() as conn:
            updated = conn.execute(
                """
                update runs
                set status = 'failed',
                    error = ?,
                    updated_at = ?
                where run_id = ?
                """,
                (error, now, run_id),
            )
            if updated.rowcount == 0:
                raise KeyError(f"run not found: {run_id}")

    def count_steps(self, run_id: str) -> int:
        return self._count(run_id=run_id, table="steps")

    def count_attempts(self, run_id: str) -> int:
        return self._count(run_id=run_id, table="attempts")

    def count_tool_calls(self, run_id: str) -> int:
        return self._count(run_id=run_id, table="tool_calls")

    def count_retries(self, run_id: str) -> int:
        self.get_run(run_id)
        with self._connect() as conn:
            row = conn.execute(
                "select count(*) as count from attempts where run_id = ? and attempt_index > 0",
                (run_id,),
            ).fetchone()
        return int(row["count"])

    def count_failed_attempts(self, run_id: str) -> int:
        self.get_run(run_id)
        with self._connect() as conn:
            row = conn.execute(
                "select count(*) as count from attempts where run_id = ? and status = 'failed'",
                (run_id,),
            ).fetchone()
        return int(row["count"])

    def last_failed_attempt(self, run_id: str) -> StoredAttempt | None:
        self.get_run(run_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                select run_id, step_index, attempt_index, input_hash, status,
                       outcome_kind, output_json, error, error_category,
                       is_retryable, started_at, finished_at
                from attempts
                where run_id = ? and status = 'failed'
                order by step_index desc, attempt_index desc
                limit 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return self._stored_attempt(row)

    def has_step_gaps(self, run_id: str) -> bool:
        self.get_run(run_id)
        with self._connect() as conn:
            row = conn.execute(
                """
                select count(*) as count, min(step_index) as min_index, max(step_index) as max_index
                from steps
                where run_id = ?
                """,
                (run_id,),
            ).fetchone()
        count = int(row["count"])
        if count == 0:
            return True
        return int(row["min_index"]) != 0 or int(row["max_index"]) != count - 1

    def has_attempt_gaps(self, run_id: str) -> bool:
        self.get_run(run_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                select step_index, count(*) as count,
                       min(attempt_index) as min_index,
                       max(attempt_index) as max_index,
                       count(distinct input_hash) as input_hash_count
                from attempts
                where run_id = ?
                group by step_index
                """,
                (run_id,),
            ).fetchall()
        for row in rows:
            count = int(row["count"])
            if count == 0:
                continue
            if int(row["min_index"]) != 0 or int(row["max_index"]) != count - 1:
                return True
            if int(row["input_hash_count"]) > 1:
                return True
        return False

    def _update_status(self, *, run_id: str, status: str) -> None:
        now = _now()
        with self._connect() as conn:
            updated = conn.execute(
                """
                update runs
                set status = ?,
                    updated_at = ?
                where run_id = ?
                """,
                (status, now, run_id),
            )
            if updated.rowcount == 0:
                raise KeyError(f"run not found: {run_id}")

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("pragma journal_mode = wal")
            self._conn.execute("pragma synchronous = normal")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _save_step_on_connection(
        self,
        *,
        conn: sqlite3.Connection,
        run_id: str,
        step_index: int,
        state_json: str,
        now: str,
    ) -> None:
        conn.execute(
            """
            insert into steps(run_id, step_index, state_json, created_at)
            values (?, ?, ?, ?)
            """,
            (run_id, step_index, state_json, now),
        )
        self._update_run_state_on_connection(
            conn=conn,
            run_id=run_id,
            step_index=step_index,
            state_json=state_json,
            now=now,
        )

    def _update_run_state_on_connection(
        self,
        *,
        conn: sqlite3.Connection,
        run_id: str,
        step_index: int,
        state_json: str,
        now: str,
    ) -> None:
        updated = conn.execute(
            """
            update runs
            set status = 'running',
                step_index = ?,
                state_json = ?,
                result_json = null,
                error = null,
                updated_at = ?
            where run_id = ?
            """,
            (step_index, state_json, now, run_id),
        )
        if updated.rowcount == 0:
            raise KeyError(f"run not found: {run_id}")

    def _mark_completed_on_connection(
        self,
        *,
        conn: sqlite3.Connection,
        run_id: str,
        step_index: int,
        result_json: str,
        now: str,
    ) -> None:
        updated = conn.execute(
            """
            update runs
            set status = 'completed',
                step_index = ?,
                state_json = null,
                result_json = ?,
                error = null,
                updated_at = ?
            where run_id = ?
            """,
            (step_index, result_json, now, run_id),
        )
        if updated.rowcount == 0:
            raise KeyError(f"run not found: {run_id}")

    def _stored_tool_call(self, row: sqlite3.Row) -> StoredToolCall:
        return StoredToolCall(
            tool_call_id=str(row["tool_call_id"]),
            run_id=str(row["run_id"]),
            step_index=int(row["step_index"]),
            attempt_index=int(row["attempt_index"]),
            idempotency_key=str(row["idempotency_key"]),
            tool_name=str(row["tool_name"]),
            request_hash=str(row["request_hash"]),
            status=str(row["status"]),
            result_json=row["result_json"],
            error=row["error"],
            error_category=row["error_category"],
            started_at=str(row["started_at"]),
            finished_at=row["finished_at"],
        )

    def _stored_attempt(self, row: sqlite3.Row) -> StoredAttempt:
        return StoredAttempt(
            run_id=str(row["run_id"]),
            step_index=int(row["step_index"]),
            attempt_index=int(row["attempt_index"]),
            input_hash=str(row["input_hash"]),
            status=str(row["status"]),
            outcome_kind=row["outcome_kind"],
            output_json=row["output_json"],
            error=row["error"],
            error_category=row["error_category"],
            is_retryable=None if row["is_retryable"] is None else bool(row["is_retryable"]),
            started_at=str(row["started_at"]),
            finished_at=row["finished_at"],
        )

    def _count(self, *, run_id: str, table: str) -> int:
        if table not in {"steps", "attempts", "tool_calls"}:
            raise ValueError(f"unsupported count table: {table}")
        self.get_run(run_id)
        with self._connect() as conn:
            row = conn.execute(
                f"select count(*) as count from {table} where run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["count"])

    def _init_db(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists runs(
                    run_id text primary key,
                    status text not null,
                    step_index integer not null,
                    state_json text,
                    result_json text,
                    error text,
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            conn.execute(
                """
                create table if not exists steps(
                    run_id text not null,
                    step_index integer not null,
                    state_json text not null,
                    created_at text not null,
                    primary key(run_id, step_index)
                )
                """
            )
            conn.execute(
                """
                create table if not exists committed_steps(
                    run_id text not null,
                    step_index integer not null,
                    input_hash text not null,
                    outcome_kind text not null,
                    state_json text,
                    result_json text,
                    created_at text not null,
                    primary key(run_id, step_index),
                    unique(run_id, step_index, input_hash)
                )
                """
            )
            conn.execute(
                """
                create table if not exists attempts(
                    run_id text not null,
                    step_index integer not null,
                    attempt_index integer not null,
                    input_hash text not null,
                    status text not null,
                    outcome_kind text,
                    output_json text,
                    error text,
                    error_category text,
                    is_retryable integer,
                    started_at text not null,
                    finished_at text,
                    primary key(run_id, step_index, attempt_index)
                )
                """
            )
            conn.execute(
                """
                create table if not exists tool_calls(
                    tool_call_id text primary key,
                    run_id text not null,
                    step_index integer not null,
                    attempt_index integer not null,
                    idempotency_key text not null,
                    tool_name text not null,
                    request_hash text not null,
                    status text not null,
                    result_json text,
                    error text,
                    error_category text,
                    started_at text not null,
                    finished_at text
                )
                """
            )
            conn.execute(
                """
                create index if not exists idx_tool_calls_idempotency_key
                on tool_calls(idempotency_key, status)
                """
            )
            conn.execute(
                """
                create index if not exists idx_tool_calls_run_step_attempt
                on tool_calls(run_id, step_index, attempt_index)
                """
            )
            self._migrate_schema(conn)

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        version = int(conn.execute("pragma user_version").fetchone()[0])
        if version < _SCHEMA_VERSION:
            conn.execute("drop index if exists idx_steps_run_step")
            conn.execute("drop index if exists idx_attempts_run_step_attempt")
            conn.execute(f"pragma user_version = {_SCHEMA_VERSION}")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require_json(value: str | None, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required")
    return value


def _limit_clause(*, limit: int | None, offset: int) -> tuple[str, tuple[int, ...]]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if limit is None:
        if offset == 0:
            return "", ()
        return " limit -1 offset ?", (offset,)
    if limit < 0:
        raise ValueError("limit must be >= 0")
    return " limit ? offset ?", (limit, offset)
