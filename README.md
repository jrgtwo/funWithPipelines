# A simple cli interface to test different local models

## Add models to models folder
Create a `/models` folder to add your local models to 

### create virtual env
`python -m venv .env`

### install dependencies
`pip install -r requirements.txt`


## MCP Server (`src/mcp/llm_server.py`)

Exposes a local HuggingFace model as an [MCP](https://modelcontextprotocol.io) server with three tools (`generate`, `chat`, `get_weather`) and one resource (`llm://info`).

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

| Tool | Required | Optional | Description |
|------|----------|----------|-------------|
| `generate` | `prompt` | see below | Raw text completion — returns only the newly generated tokens. |
| `chat` | `messages` | see below | Chat-style completion. `messages` is a list of `{"role": "...", "content": "..."}` objects with roles `system`, `user`, or `assistant`. |
| `get_weather` | `location` | `units` | Current weather for any city or region via [Open-Meteo](https://open-meteo.com). No API key required. |

`generate` and `chat` accept the same optional generation parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | `512` | Maximum number of tokens to generate. |
| `temperature` | `0.7` | Sampling temperature. `0` = greedy (deterministic). Higher values increase randomness. |
| `top_p` | `0.9` | Nucleus sampling — only tokens whose cumulative probability reaches `top_p` are considered. |
| `top_k` | `0` | Limits sampling to the top-k most likely tokens. `0` = disabled. Combine with `top_p` for finer control. |
| `repetition_penalty` | `1.0` | Penalises tokens that have already appeared. `1.0` = no penalty. Values like `1.1`–`1.3` noticeably reduce looping. |
| `stop_sequences` | `null` | List of strings that immediately halt generation when produced (e.g. `["###", "User:"]`). |
| `seed` | `null` | Integer RNG seed for reproducible outputs. Same seed + same inputs = same output. |

`get_weather` accepts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `location` | *(required)* | City name or region to look up (e.g. `"London"`, `"New York"`, `"Tokyo"`). |
| `units` | `"metric"` | `"metric"` for °C / km/h, or `"imperial"` for °F / mph. |

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

        # Basic generate call
        result = await client.call_tool(
            "generate",
            {"prompt": "The capital of France is", "max_new_tokens": 20},
        )
        print(result[0].text)

        # Reproducible output with a seed
        result = await client.call_tool(
            "generate",
            {"prompt": "Once upon a time", "max_new_tokens": 50, "seed": 42},
        )
        print(result[0].text)

        # Reduce repetition and stop on a sentinel string
        result = await client.call_tool(
            "generate",
            {
                "prompt": "List three facts about the moon:",
                "max_new_tokens": 200,
                "repetition_penalty": 1.2,
                "stop_sequences": ["4."],
            },
        )
        print(result[0].text)

        # Chat with tighter sampling (top_k + top_p together)
        result = await client.call_tool(
            "chat",
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
                "temperature": 0.5,
                "top_k": 50,
                "top_p": 0.9,
            },
        )
        print(result[0].text)

        # Current weather (metric)
        result = await client.call_tool(
            "get_weather",
            {"location": "Tokyo"},
        )
        print(result[0].text)

        # Current weather (imperial)
        result = await client.call_tool(
            "get_weather",
            {"location": "New York", "units": "imperial"},
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
# generate (basic)
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

# generate (with repetition penalty, stop sequence, and seed)
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "generate",
      "arguments": {
        "prompt": "List three facts about the moon:",
        "max_new_tokens": 200,
        "repetition_penalty": 1.2,
        "stop_sequences": ["4."],
        "seed": 42
      }
    }
  }'

# chat (with top_k + top_p sampling)
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "chat",
      "arguments": {
        "messages": [
          { "role": "system", "content": "You are a helpful assistant." },
          { "role": "user",   "content": "What is 2 + 2?" }
        ],
        "temperature": 0.5,
        "top_k": 50,
        "top_p": 0.9
      }
    }
  }'

# get_weather (metric, default)
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{
    "jsonrpc": "2.0",
    "id": 5,
    "method": "tools/call",
    "params": {
      "name": "get_weather",
      "arguments": { "location": "Tokyo" }
    }
  }'

# get_weather (imperial)
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{
    "jsonrpc": "2.0",
    "id": 6,
    "method": "tools/call",
    "params": {
      "name": "get_weather",
      "arguments": { "location": "New York", "units": "imperial" }
    }
  }'
```

**Step 4 — End the session**

```bash
curl -s -X DELETE http://127.0.0.1:8000/mcp \
  -H "Mcp-Session-Id: $SESSION"
```
