from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any


ToolFn = Callable[..., Awaitable[Any] | Any]


class ToolRegistry:
    """Minimal async-friendly tool registry."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn) -> None:
        if not name:
            raise ValueError("tool name must not be empty")
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = fn

    def tool(self, name: str) -> Callable[[ToolFn], ToolFn]:
        def decorator(fn: ToolFn) -> ToolFn:
            self.register(name, fn)
            return fn

        return decorator

    async def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            fn = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool not found: {name}") from exc

        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def names(self) -> list[str]:
        return sorted(self._tools)
