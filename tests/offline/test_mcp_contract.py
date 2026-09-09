"""Contract test for the MCP tool server.

The schemas here live in another process and could live in another team's
repo. Unlike TOOL_SCHEMA, this side is not ours to keep in sync - so the
assertion is 'does the server still offer what we expect', the same shape of
test you would write against a REST API you do not own.

No model calls: this is free and belongs in the offline tier.
"""

import asyncio

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import sys

SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "llm_testing.orders_mcp_server"],
)


async def _list_tools():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return (await session.list_tools()).tools


@pytest.fixture(scope="module")
def tools():
    return asyncio.run(_list_tools())


def test_server_exposes_the_expected_tools(tools):
    assert {t.name for t in tools} == {"get_order_status"}


def test_tool_still_takes_the_argument_we_send(tools):
    """A renamed or newly-required field is a silent break: our agent keeps
    sending the old shape and the server rejects or misreads it."""
    schema = next(t for t in tools if t.name == "get_order_status").input_schema

    assert "order_id" in schema["properties"], f"order_id gone: {schema}"
    assert schema["required"] == ["order_id"], f"required fields changed: {schema}"


def test_every_tool_has_a_description(tools):
    """The description is the only text the model uses to choose a tool. An
    empty one on a remote tool means silent mis-selection."""
    for t in tools:
        assert t.description, f"{t.name} has no description"