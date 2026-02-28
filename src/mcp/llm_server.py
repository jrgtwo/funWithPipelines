"""
FastMCP server exposing a local HuggingFace LLM as MCP tools.

Usage:
    python src/llm_server.py --model models/Qwen2.5-7B-Instruct

Tools exposed:
    generate(prompt, ...)            - raw text completion
    chat(messages, ...)              - chat-style with role/content messages
    get_weather(location, units)     - current weather via Open-Meteo (no key needed)

Resources:
    llm://info               - model metadata
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# CLI args  (parsed before anything else)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="FastMCP local-LLM server")
_parser.add_argument("--model", required=True, help="Path to local model directory")
_parser.add_argument(
    "--transport",
    default="stdio",
    choices=["stdio", "http"],
    help="Transport to use (default: stdio)",
)
_parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
_args = _parser.parse_args()


# ---------------------------------------------------------------------------
# Logging — always to stderr so stdout stays clean for MCP protocol
# ---------------------------------------------------------------------------
def _log(msg: str) -> None:
    print(f"[llm-server] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Model state  (populated in lifespan, consumed by tools)
# ---------------------------------------------------------------------------
class _State:
    model: AutoModelForCausalLM | None = None
    tokenizer: AutoTokenizer | None = None
    model_path: str = _args.model
    device: str = "cpu"


_state = _State()


# ---------------------------------------------------------------------------
# Lifespan — load once on startup, clean up on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(server: FastMCP):
    _log(f"Loading model from '{_state.model_path}' ...")
    _state.device = "cuda" if torch.cuda.is_available() else "cpu"

    _state.tokenizer = AutoTokenizer.from_pretrained(_state.model_path)
    _state.model = AutoModelForCausalLM.from_pretrained(
        _state.model_path,
        torch_dtype=torch.float16 if _state.device == "cuda" else torch.float32,
        device_map=_state.device,
    )
    _state.model.eval()
    _log(f"Model ready on {_state.device}.")

    if _args.transport == "http":
        _log(f"Listening at http://localhost:{_args.port}/mcp")
    else:
        _log("Listening on stdio.")

    yield  # server runs here

    _log("Shutting down, releasing model.")
    del _state.model
    del _state.tokenizer
    _state.model = None
    _state.tokenizer = None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
mcp = FastMCP("local-llm", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _generate_tokens(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    stop_sequences: list[str] | None = None,
    seed: int | None = None,
) -> str:
    """Tokenise, run model.generate, decode only the new tokens."""
    model = _state.model
    tokenizer = _state.tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("Model is not loaded.")

    if seed is not None:
        torch.manual_seed(seed)
        if _state.device == "cuda":
            torch.cuda.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        if top_k > 0:
            gen_kwargs["top_k"] = top_k
    if repetition_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = repetition_penalty
    if stop_sequences:
        gen_kwargs["stop_strings"] = stop_sequences
        gen_kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0][input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def _build_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Apply the tokenizer's chat template, or fall back to a plain format."""
    tokenizer = _state.tokenizer
    if tokenizer is None:
        raise RuntimeError("Model is not loaded.")

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback: simple "role: content" lines
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    lines.append("assistant:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def generate(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    stop_sequences: list[str] | None = None,
    seed: int | None = None,
) -> str:
    """
    Generate text from a raw prompt using the local LLM.

    Args:
        prompt:             The input text to continue.
        max_new_tokens:     Maximum tokens to generate (default 512).
        temperature:        Sampling temperature; 0 = greedy (default 0.7).
        top_p:              Nucleus-sampling probability (default 0.9).
        top_k:              Top-k vocabulary filtering; 0 = disabled (default 0).
        repetition_penalty: Penalty for repeating tokens; 1.0 = no penalty (default 1.0).
        stop_sequences:     List of strings that halt generation when produced.
        seed:               RNG seed for reproducible outputs.

    Returns:
        The generated text (input prompt NOT included).
    """
    return _generate_tokens(
        prompt, max_new_tokens, temperature, top_p,
        top_k, repetition_penalty, stop_sequences, seed,
    )


@mcp.tool()
def chat(
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    stop_sequences: list[str] | None = None,
    seed: int | None = None,
) -> str:
    """
    Chat with the local LLM using a conversation history.

    Args:
        messages:           List of {"role": "...", "content": "..."} dicts.
                            Roles: "system", "user", "assistant".
        max_new_tokens:     Maximum tokens to generate (default 512).
        temperature:        Sampling temperature; 0 = greedy (default 0.7).
        top_p:              Nucleus-sampling probability (default 0.9).
        top_k:              Top-k vocabulary filtering; 0 = disabled (default 0).
        repetition_penalty: Penalty for repeating tokens; 1.0 = no penalty (default 1.0).
        stop_sequences:     List of strings that halt generation when produced.
        seed:               RNG seed for reproducible outputs.

    Returns:
        The assistant's reply as plain text.
    """
    prompt = _build_chat_prompt(messages)
    return _generate_tokens(
        prompt, max_new_tokens, temperature, top_p,
        top_k, repetition_penalty, stop_sequences, seed,
    )


# ---------------------------------------------------------------------------
# Weather tool  (Open-Meteo — free, no API key)
# ---------------------------------------------------------------------------
_WMO_CODES: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm w/ slight hail", 99: "Thunderstorm w/ heavy hail",
}


@mcp.tool()
async def get_weather(location: str, units: str = "metric") -> str:
    """
    Fetch the current weather for a location using the free Open-Meteo API.
    No API key required.

    Args:
        location: City name or region (e.g. "London", "New York", "Tokyo").
        units:    "metric" (°C, km/h) or "imperial" (°F, mph). Default: metric.

    Returns:
        A short summary of current conditions (temperature, humidity,
        wind speed, and weather description).
    """
    if units not in ("metric", "imperial"):
        return f"Invalid units '{units}'. Choose 'metric' or 'imperial'."

    async with httpx.AsyncClient(timeout=10) as client:
        # 1. Geocode the location name → lat/lon
        geo_resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
        )
        geo_resp.raise_for_status()
        geo = geo_resp.json()

        results = geo.get("results")
        if not results:
            return f"Location '{location}' not found."

        r = results[0]
        lat, lon = r["latitude"], r["longitude"]
        display_name = ", ".join(
            filter(None, [r.get("name"), r.get("admin1"), r.get("country")])
        )

        # 2. Fetch current weather
        temp_unit = "celsius" if units == "metric" else "fahrenheit"
        wind_unit = "kmh" if units == "metric" else "mph"
        wx_resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "temperature_unit": temp_unit,
                "wind_speed_unit": wind_unit,
            },
        )
        wx_resp.raise_for_status()
        wx = wx_resp.json()

    current = wx["current"]
    temp      = current["temperature_2m"]
    humidity  = current["relative_humidity_2m"]
    wind      = current["wind_speed_10m"]
    code      = current["weather_code"]
    description = _WMO_CODES.get(code, f"Unknown (WMO {code})")

    t_sym = "°C" if units == "metric" else "°F"
    w_sym = "km/h" if units == "metric" else "mph"

    return (
        f"Weather in {display_name}:\n"
        f"  Conditions:  {description}\n"
        f"  Temperature: {temp}{t_sym}\n"
        f"  Humidity:    {humidity}%\n"
        f"  Wind:        {wind} {w_sym}"
    )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------
@mcp.resource("llm://info")
def model_info() -> str:
    """Metadata about the currently loaded model."""
    if _state.model is None:
        return "Model not loaded."

    param = next(_state.model.parameters())
    total_params = sum(p.numel() for p in _state.model.parameters())
    return "\n".join([
        f"path:       {_state.model_path}",
        f"device:     {_state.device}",
        f"dtype:      {param.dtype}",
        f"parameters: {total_params / 1e9:.2f}B",
    ])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if _args.transport == "http":
        from starlette.middleware import Middleware
        from starlette.middleware.cors import CORSMiddleware

        _log(f"Starting HTTP transport on port {_args.port}.")
        mcp.run(
            "http",
            port=_args.port,
            show_banner=False,
            middleware=[
                Middleware(
                    CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"],
                    expose_headers=["Mcp-Session-Id"],
                )
            ],
        )
    else:
        mcp.run("stdio", show_banner=False)
