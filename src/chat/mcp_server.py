"""Standalone FastMCP server — exposes the local HuggingFace model as a tool.

Run as:
    python -m src.chat.mcp_server --model models/<name> [--quantize 4bit] [--device cuda]

Uses MCP stdio transport (the fastmcp default). stdout is owned exclusively by
the transport; all logs go to stderr.

Design: mcp.run() is called immediately so the MCP handshake completes at once.
The model loads in a background thread; generate() blocks until it is ready.
"""
import argparse
import json
import sys
import threading
import types

from fastmcp import FastMCP

# Module-level state — populated by the background loader in __main__
_model = None
_tokenizer = None
_max_tokens: int = 512
_max_context: int = 2048
_model_path: str = ""
_model_args = None
_model_ready = threading.Event()   # set() when model is loaded and ready
_model_error: str = ""             # non-empty if loading failed


# ------------------------------------------------------------------
# FastMCP app
# ------------------------------------------------------------------

mcp = FastMCP("local-llm")


@mcp.tool()
def generate(prompt: str, system_prompt: str = "", max_tokens: int = 0) -> str:
    """Generate a response from the local language model.

    Args:
        prompt: The user message to respond to.
        system_prompt: Optional system instruction that sets model behaviour.
        max_tokens: Maximum tokens to generate. 0 uses the server default.

    Returns:
        The model's text response.
    """
    # Wait up to 5 minutes for the model to finish loading
    if not _model_ready.wait(timeout=300):
        return "Error: model loading timed out (5 min). Check server stderr for details."
    if _model_error:
        return f"Error: model failed to load — {_model_error}"

    import torch
    from transformers import TextIteratorStreamer

    n_tokens = max_tokens if max_tokens > 0 else _max_tokens

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    encoded = _tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True,
    )
    input_ids = encoded["input_ids"].to(_model.device)
    attention_mask = encoded["attention_mask"].to(_model.device)

    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=n_tokens,
        pad_token_id=_tokenizer.eos_token_id,
        stop_strings=["USER:", "SYSTEM:"],
        tokenizer=_tokenizer,
        streamer=streamer,
    )

    def _gen():
        with torch.inference_mode():
            _model.generate(**gen_kwargs)

    thread = threading.Thread(target=_gen)
    thread.start()
    chunks: list[str] = []
    for text in streamer:
        chunks.append(text)
    thread.join()

    del input_ids, attention_mask
    torch.cuda.empty_cache()

    reply = "".join(chunks)
    reply = reply.replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t")
    return reply.strip()


@mcp.resource("model://info")
def model_info() -> str:
    """Returns metadata about the currently loaded model as JSON."""
    if not _model_ready.is_set():
        return json.dumps({"status": "loading", "model_path": _model_path})
    if _model_error:
        return json.dumps({"status": "error", "error": _model_error})
    return json.dumps(
        {
            "status": "loaded",
            "model_path": _model_path,
            "quantize": _model_args.quantize if _model_args else None,
            "device": str(next(_model.parameters()).device),
            "max_tokens": _max_tokens,
            "max_context": _max_context,
            "model_type": getattr(_model.config, "model_type", "unknown"),
        },
        indent=2,
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Run a local HuggingFace model as a FastMCP server")
    parser.add_argument("--model", required=True, help="Path to the model directory")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--device", default="cuda", help="Device map strategy (cuda, auto, cpu)")
    parser.add_argument("--max-tokens", type=int, default=None)
    return parser.parse_args()


def _load_model_background(model_path, model_args):
    """Load model in a daemon thread; signals _model_ready when done."""
    global _model, _tokenizer, _max_tokens, _max_context, _model_error
    try:
        from .model import load_model
        _model, _tokenizer, _max_tokens, _max_context = load_model(model_path, model_args)
        print("[mcp_server] Model ready.", file=sys.stderr)
    except Exception as exc:
        _model_error = str(exc)
        print(f"[mcp_server] Model loading failed: {exc}", file=sys.stderr)
    finally:
        _model_ready.set()  # unblock generate() either way


if __name__ == "__main__":
    cli_args = _parse_args()

    # Redirect the shared Rich console to stderr BEFORE importing model.py.
    # model.py does `from .ui import console` at import time; patching first
    # ensures it binds to the stderr console and never writes to stdout.
    from rich.console import Console as _StderrConsole
    from . import ui as _ui_mod
    _ui_mod.console = _StderrConsole(stderr=True, markup=True, highlight=False)

    _model_path = cli_args.model
    _model_args = types.SimpleNamespace(
        quantize=cli_args.quantize,
        device=cli_args.device,
        max_tokens=cli_args.max_tokens,
    )

    # Start model loading in a daemon background thread so mcp.run() can
    # begin immediately and complete the MCP handshake without waiting.
    print(f"[mcp_server] Loading model in background: {_model_path}", file=sys.stderr)
    loader = threading.Thread(
        target=_load_model_background,
        args=(_model_path, _model_args),
        daemon=True,
    )
    loader.start()

    print("[mcp_server] Starting MCP server (stdio)...", file=sys.stderr)
    mcp.run()  # stdio is the fastmcp default — reads stdin, writes stdout
