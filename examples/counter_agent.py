from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom_agent import AgentRunner, Complete, Continue, RunContext, SQLiteCheckpointStore


@dataclass(frozen=True)
class CounterState:
    current: int
    target: int


async def step(state: CounterState, ctx: RunContext) -> Continue[CounterState] | Complete[dict[str, int]]:
    await asyncio.sleep(0)
    if state.current >= state.target:
        return Complete({"current": state.current, "steps": ctx.step_index})
    return Continue(CounterState(current=state.current + 1, target=state.target))


def encode_state(state: CounterState) -> dict[str, int]:
    return {"current": state.current, "target": state.target}


def decode_state(data: Any) -> CounterState:
    return CounterState(current=int(data["current"]), target=int(data["target"]))


def encode_result(result: dict[str, int]) -> dict[str, int]:
    return result


def decode_result(data: Any) -> dict[str, int]:
    return {"current": int(data["current"]), "steps": int(data["steps"])}


def build_runner(db_path: str) -> AgentRunner[CounterState, dict[str, int]]:
    return AgentRunner(
        step=step,
        store=SQLiteCheckpointStore(db_path),
        encode_state=encode_state,
        decode_state=decode_state,
        encode_result=encode_result,
        decode_result=decode_result,
    )


def initial_state() -> CounterState:
    return CounterState(current=0, target=10)


async def _demo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "runs.sqlite")
        runner = build_runner(db)
        first = await runner.start(run_id="demo", initial_state=initial_state(), max_steps=5)
        print(first)
        second = await runner.resume(run_id="demo", max_steps=100)
        print(second)


if __name__ == "__main__":
    asyncio.run(_demo())
