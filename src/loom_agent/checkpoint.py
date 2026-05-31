from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class StoredRun:
    run_id: str
    status: str
    step_index: int
    state_json: str | None
    result_json: str | None
    error: str | None


class SQLiteCheckpointStore:
    """SQLite-backed checkpoint store for JSON-encoded run state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
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

    def save_step(self, *, run_id: str, step_index: int, state_json: str) -> None:
        now = _now()
        with self._connect() as conn:
            conn.execute(
                """
                insert into steps(run_id, step_index, state_json, created_at)
                values (?, ?, ?, ?)
                """,
                (run_id, step_index, state_json, now),
            )
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

    def mark_paused(self, *, run_id: str) -> None:
        self._update_status(run_id=run_id, status="paused")

    def mark_completed(self, *, run_id: str, step_index: int, result_json: str) -> None:
        now = _now()
        with self._connect() as conn:
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
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

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


def _now() -> str:
    return datetime.now(UTC).isoformat()
