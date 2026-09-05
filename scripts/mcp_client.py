"""Small synchronous bridge for configured MCP servers."""
from __future__ import annotations
import asyncio, json, threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Mapping

class MCPUnavailableError(RuntimeError):
    pass

def _run(coro):
    try: asyncio.get_running_loop()
    except RuntimeError: return asyncio.run(coro)
    result, error = [], []
    def worker():
        try: result.append(asyncio.run(coro))
        except BaseException as exc: error.append(exc)
    thread = threading.Thread(target=worker, daemon=True); thread.start(); thread.join()
    if error: raise error[0]
    return result[0] if result else None

class MCPToolRegistry:
    def __init__(self, servers: Mapping[str, Mapping[str, Any]] | None = None, *, config_path: str | Path | None = None, max_result_chars: int = 12000):
        if config_path:
            path = Path(config_path)
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8")); servers = loaded.get("mcpServers", loaded)
        self.servers = dict(servers or {}); self.max_result_chars = max(1000, int(max_result_chars))

    @property
    def enabled(self) -> bool:
        return bool(self.servers)

    @staticmethod
    def _adapter():
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            return MultiServerMCPClient
        except ImportError: return None

    @staticmethod
    def _raw_parts():
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
            return ClientSession, StdioServerParameters, stdio_client
        except ImportError as exc: raise MCPUnavailableError("MCP 未安装，请执行: python -m pip install mcp") from exc

    async def _adapter_tools(self):
        return await self._adapter()(self.servers).get_tools()

    async def _raw_session(self, stack, config):
        Session, Params, stdio = self._raw_parts()
        params = Params(command=str(config["command"]), args=[str(x) for x in config.get("args", [])], env=config.get("env"), cwd=config.get("cwd"))
        read, write = await stack.enter_async_context(stdio(params)); session = await stack.enter_async_context(Session(read, write)); await session.initialize(); return session

    async def _raw_list(self):
        result = []
        async with AsyncExitStack() as stack:
            for server, config in self.servers.items():
                if config.get("transport", "stdio") != "stdio": continue
                session = await self._raw_session(stack, config)
                for tool in (await session.list_tools()).tools:
                    result.append({"name": tool.name, "description": tool.description or "", "input_schema": tool.inputSchema, "_server": server})
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        if not self.servers: return []
        tools = _run(self._adapter_tools()) if self._adapter() else _run(self._raw_list()); result = []
        for tool in tools or []:
            if isinstance(tool, dict): result.append({k: v for k, v in tool.items() if not k.startswith("_")}); continue
            schema = getattr(tool, "args_schema", None)
            result.append({"name": getattr(tool, "name", ""), "description": getattr(tool, "description", ""), "input_schema": schema.model_json_schema() if hasattr(schema, "model_json_schema") else None})
        return result

    async def _raw_call(self, name, arguments):
        async with AsyncExitStack() as stack:
            for server, config in self.servers.items():
                if config.get("transport", "stdio") != "stdio": continue
                session = await self._raw_session(stack, config); names = {tool.name for tool in (await session.list_tools()).tools}
                candidate = name[len(server) + 1:] if name.startswith(server + "_") else name
                if candidate in names:
                    response = await session.call_tool(candidate, dict(arguments))
                    texts = [getattr(item, "text", "") for item in (response.content or [])]
                    if any(x for x in texts):
                        return "\n".join(x for x in texts if x)
                    # Keep an empty/structured MCP response JSON-serializable.
                    return response.model_dump() if hasattr(response, "model_dump") else response
        raise KeyError(f"MCP tool not found: {name}")

    async def _call(self, name, arguments):
        if not self._adapter(): return await self._raw_call(name, arguments)
        for tool in await self._adapter_tools():
            if getattr(tool, "name", None) == name:
                return await tool.ainvoke(dict(arguments)) if hasattr(tool, "ainvoke") else tool.invoke(dict(arguments))
        raise KeyError(f"MCP tool not found: {name}")

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> Any:
        value = _run(self._call(name, arguments or {}))
        if isinstance(value, str) and len(value) > self.max_result_chars: return value[: self.max_result_chars] + "\n[结果已截断]"
        return value

def load_mcp_registry(config_path: str | Path = "configs/mcp_servers.json") -> MCPToolRegistry:
    return MCPToolRegistry(config_path=config_path)
