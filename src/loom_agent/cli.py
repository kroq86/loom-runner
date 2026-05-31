from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from flow_xray import trace


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    module = _load_module(args.module)
    runner = module.build_runner(args.db)

    async def execute() -> Any:
        if args.command == "run":
            return await runner.start(
                run_id=args.run_id,
                initial_state=module.initial_state(),
                max_steps=args.max_steps,
            )
        if args.command == "resume":
            return await runner.resume(run_id=args.run_id, max_steps=args.max_steps)
        if args.command == "list":
            return runner.list_runs()
        if args.command == "get":
            return runner.get_run(args.run_id)
        if args.command == "history":
            return runner.get_history(args.run_id, limit=args.limit, offset=args.offset)
        if args.command == "attempts":
            return runner.get_attempts(args.run_id, limit=args.limit, offset=args.offset)
        if args.command == "tool-calls":
            return runner.get_tool_calls(args.run_id, limit=args.limit, offset=args.offset)
        if args.command == "explain":
            return runner.explain_run(args.run_id)
        raise ValueError(f"unknown command: {args.command}")

    if getattr(args, "trace", None):
        result = trace.run(lambda: asyncio.run(execute()))
        result.to_html(args.trace)
        run_result = result.return_value
    else:
        run_result = asyncio.run(execute())

    print(json.dumps(_to_jsonable(run_result, runner=runner), sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="loom-runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        sub = subparsers.add_parser(name)
        sub.add_argument("module", help="Path to an agent module.")
        sub.add_argument("--run-id", required=True)
        sub.add_argument("--db", required=True)
        sub.add_argument("--max-steps", type=int, required=True)
        sub.add_argument("--trace")
    for name in ("list", "get", "history", "attempts", "tool-calls", "explain"):
        sub = subparsers.add_parser(name)
        sub.add_argument("module", help="Path to an agent module.")
        sub.add_argument("--db", required=True)
        if name != "list":
            sub.add_argument("--run-id", required=True)
        if name in {"history", "attempts", "tool-calls"}:
            sub.add_argument("--limit", type=int)
            sub.add_argument("--offset", type=int, default=0)
    return parser.parse_args(argv)


def _load_module(path_text: str):
    path = Path(path_text).resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _to_jsonable(value: Any, *, runner: Any) -> Any:
    if isinstance(value, list):
        return [_to_jsonable(item, runner=runner) for item in value]
    if dataclasses.is_dataclass(value):
        data = dataclasses.asdict(value)
        if "state" in data and data["state"] is not None:
            data["state"] = runner.encode_state(value.state)
        if "result" in data and data["result"] is not None:
            data["result"] = runner.encode_result(value.result)
        return data
    return value


if __name__ == "__main__":
    raise SystemExit(main())
