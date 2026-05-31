from __future__ import annotations

import argparse
import asyncio
import importlib.util
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
        return await runner.resume(run_id=args.run_id, max_steps=args.max_steps)

    if args.trace:
        result = trace.run(lambda: asyncio.run(execute()))
        result.to_html(args.trace)
        run_result = result.return_value
    else:
        run_result = asyncio.run(execute())

    print(run_result)
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


if __name__ == "__main__":
    raise SystemExit(main())
