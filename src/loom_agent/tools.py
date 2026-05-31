from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import ErrorCategory


ToolFn = Callable[..., Awaitable[Any] | Any]


@dataclass(frozen=True)
class ToolResult:
    success: bool
    is_error: bool = False
    error_category: str | None = None
    is_retryable: bool = False
    result_type: str = "generic"
    payload: Any = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    partial_results: list[Any] = field(default_factory=list)
    attempted_action: str | None = None
    suggested_next_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "is_error": self.is_error,
            "error_category": self.error_category,
            "is_retryable": self.is_retryable,
            "result_type": self.result_type,
            "payload": self.payload,
            "metadata": self.metadata,
            "partial_results": self.partial_results,
            "attempted_action": self.attempted_action,
            "suggested_next_steps": self.suggested_next_steps,
        }


class ToolPolicy:
    """Normalize raw tool returns into an agent-readable result envelope."""

    def normalize_result(self, result: Any) -> ToolResult:
        if not isinstance(result, dict) or not _looks_like_envelope(result):
            return ToolResult(success=True, payload=result)

        raw_category = result.get("error_category")
        if raw_category is None:
            error_category = None
        elif raw_category in {category.value for category in ErrorCategory}:
            error_category = str(raw_category)
        else:
            error_category = ErrorCategory.UNKNOWN.value

        return ToolResult(
            success=bool(result.get("success", False)),
            is_error=bool(result.get("is_error", False)),
            error_category=error_category,
            is_retryable=bool(result.get("is_retryable", False)),
            result_type=str(result.get("result_type", "generic")),
            payload=result.get("payload", {}),
            metadata=dict(result.get("metadata", {})),
            partial_results=list(result.get("partial_results", [])),
            attempted_action=result.get("attempted_action"),
            suggested_next_steps=list(result.get("suggested_next_steps", [])),
        )


class ToolRegistry:
    """Minimal async-friendly tool registry."""

    def __init__(self, *, policy: ToolPolicy | None = None) -> None:
        self._tools: dict[str, ToolFn] = {}
        self._policy = policy or ToolPolicy()

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
        return (await self.call_result(name, *args, **kwargs)).payload

    async def call_result(self, name: str, *args: Any, **kwargs: Any) -> ToolResult:
        try:
            fn = self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool not found: {name}") from exc

        result = fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return self._policy.normalize_result(result)

    def names(self) -> list[str]:
        return sorted(self._tools)


def _looks_like_envelope(result: dict[str, Any]) -> bool:
    envelope_keys = {
        "success",
        "is_error",
        "error_category",
        "is_retryable",
        "result_type",
        "payload",
        "metadata",
        "partial_results",
        "attempted_action",
        "suggested_next_steps",
    }
    return bool(envelope_keys.intersection(result))
