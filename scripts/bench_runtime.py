from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom_agent import (
    AgentRunner,
    CheckpointPolicy,
    Complete,
    Continue,
    RunContext,
    SQLiteCheckpointStore,
)


@dataclass(frozen=True)
class State:
    current: int
    target: int


def encode_state(state: State) -> dict[str, int]:
    return {"current": state.current, "target": state.target}


def decode_state(data: Any) -> State:
    return State(current=int(data["current"]), target=int(data["target"]))


async def step(state: State, ctx: RunContext) -> Continue[State] | Complete[dict[str, int]]:
    if state.current >= state.target:
        return Complete({"current": state.current})
    return Continue(State(state.current + 1, state.target))


def _policy_from_args(policy: str, checkpoint_every: int) -> CheckpointPolicy:
    if policy == "full":
        return CheckpointPolicy()
    if policy == "interval":
        return CheckpointPolicy(mode="interval", every=checkpoint_every)
    if policy == "compact":
        return CheckpointPolicy(mode="compact")
    raise ValueError(f"unknown policy: {policy}")


async def run_bench(
    *,
    steps: int,
    policy: str,
    checkpoint_every: int,
    db_path: Path,
) -> dict[str, Any]:
    checkpoint_policy = _policy_from_args(policy, checkpoint_every)
    store = SQLiteCheckpointStore(db_path)
    runner = AgentRunner(
        step=step,
        store=store,
        encode_state=encode_state,
        decode_state=decode_state,
        encode_result=lambda result: result,
        decode_result=lambda result: result,
        checkpoint_policy=checkpoint_policy,
    )

    tracemalloc.start()
    started = time.perf_counter()
    result = await runner.start(run_id="bench", initial_state=State(0, steps), max_steps=steps + 1)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    explain_started = time.perf_counter()
    runner.explain_run("bench")
    explain_elapsed = time.perf_counter() - explain_started

    stats = runner.get_stats("bench")
    store.close()
    return {
        "status": result.status,
        "policy": policy,
        "steps": steps,
        "checkpoint_every": checkpoint_every,
        "elapsed_seconds": round(elapsed, 4),
        "explain_seconds": round(explain_elapsed, 4),
        "peak_python_bytes": peak,
        "db_bytes": db_path.stat().st_size,
        "step_rows": stats.step_count,
        "attempt_rows": stats.attempt_count,
        "tool_call_rows": stats.tool_call_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark loom-runner local runtime overhead.")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument(
        "--policy",
        choices=("full", "interval", "compact"),
        default="full",
        help="Checkpoint policy (interval uses --checkpoint-every)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Retain every Nth step checkpoint when --policy=interval",
    )
    parser.add_argument("--db")
    args = parser.parse_args()

    if args.db:
        db_path = Path(args.db)
        result = asyncio.run(
            run_bench(
                steps=args.steps,
                policy=args.policy,
                checkpoint_every=args.checkpoint_every,
                db_path=db_path,
            )
        )
    else:
        with tempfile.TemporaryDirectory() as tmp:
            result = asyncio.run(
                run_bench(
                    steps=args.steps,
                    policy=args.policy,
                    checkpoint_every=args.checkpoint_every,
                    db_path=Path(tmp) / "bench.sqlite",
                )
            )

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
