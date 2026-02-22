# A simple cli interface to test different local models

## Add models to models folder
Create a `/models` folder to add your local models to 

### create virtual env
`python -m venv .env`

### install dependencies
`pip install -r requirements.txt`

### run
`> cd src`
`> python -m chat`

---

## MCP Server (`src/mcp/llm_server.py`)

Exposes a local HuggingFace model as an [MCP](https://modelcontextprotocol.io) server with two tools (`generate`, `chat`) and one resource (`llm://info`).

### Transports

#### stdio (default)
Used when another process (e.g. `src/chat/app.py`) spawns the server as a subprocess.

```bash
python -m src.mcp.llm_server --model models/<model-name>
```

#### HTTP
Runs a local HTTP server so any MCP-compatible client (browser app, Claude Desktop, etc.) can connect.

```bash
python -m src.mcp.llm_server --model models/<model-name> --transport http --port 8000
```

The MCP endpoint is available at `http://127.0.0.1:8000/mcp`.
CORS is enabled for all origins, so browser-based clients can POST directly.

### Tools

| Tool | Arguments | Description |
|------|-----------|-------------|
| `generate` | `prompt`, `max_new_tokens` (512), `temperature` (0.7), `top_p` (0.9) | Raw text completion — returns only the newly generated tokens. |
| `chat` | `messages`, `max_new_tokens` (512), `temperature` (0.7), `top_p` (0.9) | Chat-style completion. `messages` is a list of `{"role": "...", "content": "..."}` objects with roles `system`, `user`, or `assistant`. |

### Resources

| URI | Description |
|-----|-------------|
| `llm://info` | Returns model path, device, dtype, and parameter count. |

### Examples

#### Python (fastmcp Client — recommended)

The fastmcp `Client` handles the entire initialization and session lifecycle automatically.

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        # Session is initialized automatically on __aenter__

        # Call generate tool
        result = await client.call_tool(
            "generate",
            {"prompt": "The capital of France is", "max_new_tokens": 20},
        )
        print(result[0].text)

        # Chat-style call
        result = await client.call_tool(
            "chat",
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
            },
        )
        print(result[0].text)

        # Read model info resource
        info = await client.read_resource("llm://info")
        print(info[0].text)

    # Session is cleanly torn down on __aexit__

asyncio.run(main())
```

#### Raw HTTP (curl)

The MCP streamable-HTTP transport requires a three-step handshake before making calls. The server issues an `Mcp-Session-Id` that must be echoed on every subsequent request, and the session must be explicitly deleted when done.

**Step 1 — Initialize** (exchange capabilities, receive session ID)

```bash
curl -sD - -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": { "name": "curl-client", "version": "1.0" }
    }
  }'
# Response headers include: Mcp-Session-Id: <session-id>
# Copy that value for the steps below.
```

**Step 2 — Send `notifications/initialized`** (complete the handshake)

```bash
SESSION="<session-id from step 1>"

curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc": "2.0", "method": "notifications/initialized"}'
```

**Step 3 — Call a tool**

```bash
# generate
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "generate",
      "arguments": { "prompt": "The capital of France is", "max_new_tokens": 20 }
    }
  }'

# chat
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "chat",
      "arguments": {
        "messages": [
          { "role": "system", "content": "You are a helpful assistant." },
          { "role": "user",   "content": "What is 2 + 2?" }
        ]
      }
    }
  }'
```

**Step 4 — End the session**

```bash
curl -s -X DELETE http://127.0.0.1:8000/mcp \
  -H "Mcp-Session-Id: $SESSION"
```
