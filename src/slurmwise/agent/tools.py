from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from typing import Any, Protocol

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class ToolClient(Protocol):
    def call(self, name: str, **args: Any) -> str: ...

    def list_tools(self) -> list[dict]: ...

    def close(self) -> None: ...


def _result_text(result: Any) -> str:
    parts = []
    for c in getattr(result, "content", []) or []:
        if getattr(c, "type", "") == "text":
            parts.append(c.text)
    if getattr(result, "is_error", False):
        return "ERROR: " + ("\n".join(parts) or "tool call failed")
    return "\n".join(parts)


# calls the tool functions in-process; used by --direct and in tests (no subprocess)
class DirectToolClient:
    def __init__(self, backend=None):
        from .. import mcp_server

        self._mod = mcp_server
        if backend is not None:
            mcp_server._backend = backend

    def call(self, name: str, **args: Any) -> str:
        fn = getattr(self._mod, name)
        return str(fn(**args))

    def list_tools(self) -> list[dict]:
        tools = asyncio.run(self._mod.server.list_tools())
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]

    def close(self) -> None:
        pass


# Spawning the MCP server as a subprocess and talks to it over stdio, like any MCP client
class MCPToolClient:
    def __init__(self, command: str | None = None, args: list[str] | None = None, env: dict | None = None):
        if command is None:
            command, args = sys.executable, ["-m", "slurmwise.mcp_server"]
        self._params = StdioServerParameters(command=command, args=args or [], env={**os.environ, **(env or {})})
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True, name="slurmpilot-mcp")
        self._thread.start()
        self._ready = threading.Event()
        self._stop = None
        self._session: ClientSession | None = None
        self._err: BaseException | None = None
        asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        self._ready.wait(timeout=60)
        if self._err:
            raise RuntimeError(f"could not start MCP server: {self._err!r}")

    async def _serve(self):
        self._stop = asyncio.Event()
        try:
            async with stdio_client(self._params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except BaseException as e:
            self._err = e
            self._ready.set()

    def _run(self, coro, timeout: float = 120.0):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def call(self, name: str, **args: Any) -> str:
        assert self._session is not None
        res = self._run(self._session.call_tool(name, args))
        return _result_text(res)

    def list_tools(self) -> list[dict]:
        assert self._session is not None
        res = self._run(self._session.list_tools())
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in res.tools]

    def close(self) -> None:
        if self._stop is not None and not self._stop.is_set():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=5)
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

def get_tool_client(direct: bool = False, backend=None) -> ToolClient:
    if direct or os.environ.get("SLURMPILOT_DIRECT") == "1":
        return DirectToolClient(backend)
    return MCPToolClient()


# Wrap each MCP tool as a LangChain
def as_langchain_tools(client: ToolClient):
    from langchain_core.tools import StructuredTool

    out = []
    for t in client.list_tools():
        name = t["name"]

        def _make(n):
            def _fn(**kwargs):
                return client.call(n, **kwargs)

            return _fn

        schema = dict(t["input_schema"])
        schema.setdefault("title", name)
        out.append(StructuredTool(name=name, description=t["description"] or name, args_schema=schema, func=_make(name)))
    return out


def loads_or_text(s: str):
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s
