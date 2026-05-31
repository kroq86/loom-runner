# loom-runner

[![PyPI](https://img.shields.io/pypi/v/loom-runner)](https://pypi.org/project/loom-runner/)

Small durable checkpoint/resume runner for async state-machine loops built on
top of `loom-tailcalls` and `flow-xray`.

This is not a planner, memory system, graph DSL, hosted tracing product, or
full agent SDK. It is the first slice of a Loom-based agent runtime: run a
typed async transition loop, checkpoint each state transition, resume later,
inspect history, and explain a run.

## Loom stack

Three composable packages for **long-running async agent loops**. Each does one job; compose them as needed.

| Package | Install | Job |
| --- | --- | --- |
| **[loom-tailcalls](https://github.com/kroq86/loom-tailcalls)** | `pip install loom-tailcalls` | Write stack-safe transition loops (`@tailrec`, `@tailstream`) |
| **[flow-xray](https://github.com/kroq86/flow-xray)** | `pip install flow-xray` | Export local HTML traces (LLM/tool calls, branches, errors) |
| **[loom-runner](https://github.com/kroq86/loom-runner)** ← **this repo** | `pip install loom-runner` | Checkpoint/resume in SQLite; CLI inspect (`explain`, `history`, …) |

```text
@tailrec agent loop  →  loom-runner run/resume  →  --trace trace.html
     (shape)                  (durability)              (flow-xray)
```

**This repo** depends on `loom-tailcalls` and `flow-xray`. It adds persistence and inspection on top of stack-safe loops — not reasoning, planning, memory, or a path to AGI.

## Who it is for

- Authors of **long-running async agent loops** who need checkpoint/resume without building their own store
- Users of **[loom-tailcalls](https://github.com/kroq86/loom-tailcalls)** who want persistence and CLI inspection on top of stack-safe transitions
- Users of **[flow-xray](https://github.com/kroq86/flow-xray)** who want `--trace trace.html` from the runner CLI
- Anyone who needs an **inspectable run** (`explain`, `history`, `attempts`, `tool-calls`) rather than a black box

**Not for you** if the agent is a single LLM call, or you already have LangGraph/Temporal (or similar) with persistence you are happy with.

This is not reasoning, planning, memory, or a path to AGI — it is a **durability + observability** primitive for state-machine-shaped agent runtimes.

Runtime transitions are logged as logical steps with attempt history. A retry
does not create a new transition: for the same `run_id`, `step_index`, and
stable input hash, the runner reuses the committed outcome. Transient errors
are retryable by default; validation, business, permission, and unknown errors
fail the run unless the caller supplies a different policy.

Tool side effects are only idempotent when invoked through
`RunContext.call_tool(...)`. Direct tool calls or external effects inside a
transition are intentionally treated as unmanaged user code in this first
runtime slice.

Long runs can use bounded reads and explicit storage policies. By default the
runner keeps every checkpoint and every inline tool payload for maximum
inspectability. For larger runs, use `CheckpointPolicy(mode="interval",
every=N)` to retain only periodic history checkpoints while preserving the
current resumable state, and `PayloadPolicy(max_inline_bytes=N)` to replace
large managed tool payloads with hash/size metadata.

The import package remains `loom_agent`; the distribution and CLI are named
`loom-runner` because `loom-agent` is already occupied by an unrelated package
on PyPI.

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Minimal Shape

```python
from dataclasses import dataclass

from loom_agent import AgentRunner, Complete, Continue, RunContext, SQLiteCheckpointStore


@dataclass(frozen=True)
class State:
    current: int
    target: int


async def step(state: State, ctx: RunContext):
    if state.current >= state.target:
        return Complete({"current": state.current})
    return Continue(State(current=state.current + 1, target=state.target))


runner = AgentRunner(
    step=step,
    store=SQLiteCheckpointStore("runs.sqlite"),
    encode_state=lambda state: {"current": state.current, "target": state.target},
    decode_state=lambda data: State(**data),
    encode_result=lambda result: result,
    decode_result=lambda data: data,
)
```

## Example

```bash
loom-runner run examples/counter_agent.py --run-id demo --db runs.sqlite --max-steps 5
loom-runner resume examples/counter_agent.py --run-id demo --db runs.sqlite --max-steps 100
loom-runner list examples/counter_agent.py --db runs.sqlite
loom-runner get examples/counter_agent.py --run-id demo --db runs.sqlite
loom-runner history examples/counter_agent.py --run-id demo --db runs.sqlite
loom-runner attempts examples/counter_agent.py --run-id demo --db runs.sqlite --limit 20
loom-runner tool-calls examples/counter_agent.py --run-id demo --db runs.sqlite --limit 20
loom-runner explain examples/counter_agent.py --run-id demo --db runs.sqlite
```

Add `--trace trace.html` to either command to emit a local `flow-xray` HTML
trace. The runner traces step leaves and keeps the tail-recursive driver as the
durable loop boundary.

Or directly:

```bash
python3.13 examples/counter_agent.py
```

## Tests

```bash
python3.13 -m pytest
```

## Runtime Benchmark

```bash
python3.13 scripts/bench_runtime.py --steps 100000
python3.13 scripts/bench_runtime.py --steps 100000 --checkpoint-every 100
```

The benchmark reports wall time, retained checkpoint rows, attempt rows, DB
size, and peak Python memory. It is a local regression tool, not a hosted-scale
performance claim.
