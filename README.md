# loom-runner

Small durable checkpoint/resume runner for async state-machine loops built on
top of `loom-tailcalls` and `flow-xray`.

This is not a planner, memory system, graph DSL, hosted tracing product, or
full agent SDK. It keeps the public surface intentionally small: run a typed
async step loop, checkpoint each state transition, and resume later.

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
