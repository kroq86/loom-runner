from __future__ import annotations

import pytest

from loom_agent import ErrorCategory, ToolPolicy, ToolRegistry


@pytest.mark.asyncio
async def test_tool_registry_calls_async_and_sync_tools() -> None:
    tools = ToolRegistry()

    @tools.tool("add")
    async def add(left: int, right: int) -> int:
        return left + right

    @tools.tool("double")
    def double(value: int) -> int:
        return value * 2

    assert await tools.call("add", 2, 3) == 5
    assert await tools.call("double", 4) == 8
    assert tools.names() == ["add", "double"]


@pytest.mark.asyncio
async def test_tool_registry_missing_tool_errors() -> None:
    tools = ToolRegistry()

    with pytest.raises(KeyError, match="tool not found: missing"):
        await tools.call("missing")


def test_tool_registry_rejects_duplicate() -> None:
    tools = ToolRegistry()
    tools.register("x", lambda: 1)

    with pytest.raises(ValueError, match="tool already registered: x"):
        tools.register("x", lambda: 2)


@pytest.mark.asyncio
async def test_tool_registry_normalizes_agent_tool_result_envelope() -> None:
    tools = ToolRegistry()

    @tools.tool("lookup")
    def lookup() -> dict:
        return {
            "success": False,
            "is_error": True,
            "error_category": ErrorCategory.BUSINESS.value,
            "is_retryable": False,
            "result_type": "lookup",
            "payload": {"matches": []},
            "attempted_action": "lookup",
        }

    result = await tools.call_result("lookup")

    assert result.success is False
    assert result.is_error is True
    assert result.error_category == "business"
    assert result.payload == {"matches": []}
    assert result.attempted_action == "lookup"


def test_tool_policy_wraps_plain_result_as_success_payload() -> None:
    result = ToolPolicy().normalize_result({"plain": "value"})

    assert result.success is True
    assert result.is_error is False
    assert result.payload == {"plain": "value"}
