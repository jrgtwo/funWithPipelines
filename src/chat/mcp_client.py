"""Synchronous MCP client manager — one persistent background thread per server.

Each connected server gets its own asyncio event loop running on a daemon thread.
The subprocess is started once and kept alive for the full lifetime of the connection.
Synchronous calls bridge to the background thread via asyncio.run_coroutine_threadsafe().

This avoids the Windows-specific issue where repeated asyncio.run() calls leave the
server subprocess alive with a dangled stdin pipe that picks up terminal input.
"""
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ServerConfig:
    """Configuration for one connected MCP server."""
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None


class _PersistentConnection:
    """A long-lived MCP client connection on a dedicated daemon thread.

    The subprocess runs for the full lifetime of this object. Calls are
    submitted via asyncio.run_coroutine_threadsafe() and block until complete.
    """

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._client = None          # fastmcp Client, set in _main after connect
        self._ready = threading.Event()   # set when client is ready
        self._start_error: Exception | None = None
        self._stop_flag: asyncio.Event | None = None  # set in _main to unblock it

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background thread and block until the MCP handshake completes."""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"mcp-{self._config.name}",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._start_error:
            raise self._start_error

    def stop(self) -> None:
        """Signal the background thread to disconnect and wait for it to exit."""
        if self._loop and not self._loop.is_closed() and self._stop_flag is not None:
            self._loop.call_soon_threadsafe(self._stop_flag.set)
        if self._thread:
            self._thread.join(timeout=15)

    # ------------------------------------------------------------------
    # Synchronous call bridge
    # ------------------------------------------------------------------

    def list_tools(self):
        return self._submit(self._client.list_tools(), timeout=30)

    def call_tool(self, name: str, arguments: dict) -> list:
        return self._submit(self._client.call_tool(name, arguments), timeout=300)

    def read_resource(self, uri: str) -> list:
        return self._submit(self._client.read_resource(uri), timeout=30)

    def _submit(self, coro, timeout: float):
        """Submit a coroutine to the background event loop and block for the result."""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("Connection is not running.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except Exception as exc:
            if not self._ready.is_set():
                self._start_error = exc
        finally:
            if not self._ready.is_set():
                self._ready.set()  # unblock start() on error
            try:
                self._loop.close()
            except Exception:
                pass

    async def _main(self) -> None:
        from fastmcp import Client
        from fastmcp.client.transports import StdioTransport

        transport = StdioTransport(
            command=self._config.command,
            args=self._config.args,
            env=self._config.env,
            keep_alive=False,   # ensure subprocess is killed on disconnect
        )
        async with Client(transport) as client:
            self._client = client
            self._stop_flag = asyncio.Event()
            self._ready.set()          # unblock start()
            await self._stop_flag.wait()  # keep the connection alive


# ------------------------------------------------------------------
# Public manager
# ------------------------------------------------------------------

class MCPClientManager:
    """Manages named connections to multiple MCP servers.

    Each server gets a persistent subprocess via a background daemon thread.
    """

    def __init__(self) -> None:
        self._configs: dict[str, ServerConfig] = {}
        self._conns: dict[str, _PersistentConnection] = {}

    # --- server lifecycle ---

    def connect(self, name: str, command: str, args: list[str]) -> None:
        """Start a connection to an MCP server, validated by a list_tools probe."""
        if name in self._conns:
            raise ValueError(f"Server '{name}' is already connected. Disconnect first.")
        config = ServerConfig(name=name, command=command, args=args)
        conn = _PersistentConnection(config)
        try:
            conn.start()
        except Exception as exc:
            raise ConnectionError(f"Could not connect to '{name}': {exc}") from exc
        self._configs[name] = config
        self._conns[name] = conn

    def disconnect(self, name: str) -> None:
        """Stop the connection and kill the server subprocess."""
        if name not in self._conns:
            raise KeyError(f"No server named '{name}'.")
        self._conns.pop(name).stop()
        del self._configs[name]

    def list_servers(self) -> list[str]:
        return list(self._conns.keys())

    # --- tool operations ---

    def list_tools(self) -> dict[str, list[dict]]:
        """Return {server_name: [{name, description, schema}, ...]}."""
        result: dict[str, list[dict]] = {}
        for name, conn in self._conns.items():
            try:
                raw = conn.list_tools()
                result[name] = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "schema": t.inputSchema,
                    }
                    for t in raw
                ]
            except Exception as exc:
                result[name] = [{"error": str(exc)}]
        return result

    def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool and return the result as a plain string."""
        if server_name not in self._conns:
            raise KeyError(f"No server named '{server_name}'.")
        raw = self._conns[server_name].call_tool(tool_name, arguments)
        return _extract_text(raw)

    def fetch_resource(self, uri: str) -> str:
        """Fetch a resource URI from whichever connected server provides it."""
        errors: list[str] = []
        for name, conn in self._conns.items():
            try:
                raw = conn.read_resource(uri)
                return _extract_text(raw)
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        raise RuntimeError(
            f"No server could provide '{uri}'. Errors: {'; '.join(errors)}"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_text(results) -> str:
    """Flatten MCP content items to a plain string."""
    if not results:
        return ""
    parts: list[str] = []
    for item in results:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "blob"):
            parts.append(f"[binary data: {len(item.blob)} bytes]")
        else:
            parts.append(str(item))
    return "\n".join(parts)


def format_mcp_context(server_name: str, tool_name: str, result_text: str) -> str:
    """Format a tool result as a context block for prompt injection."""
    return f"### MCP tool result from {server_name}.{tool_name}:\n{result_text}"


def format_resource_context(uri: str, result_text: str) -> str:
    """Format a resource fetch result as a context block for prompt injection."""
    return f"### MCP resource {uri}:\n{result_text}"
