"""File reading, attaching, and input parsing with /file commands."""

from pathlib import Path
from rich.panel import Panel
from rich.syntax import Syntax

from .config import SUFFIX_TO_LANG
from .ui import console


def read_file(path_str):
    """Read a file and return (content, language) or (None, error_message)."""
    path = Path(path_str.strip())
    if not path.is_file():
        return None, f"File not found: {path}"
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, f"Cannot read binary file: {path}"
    lang = SUFFIX_TO_LANG.get(path.suffix.lower(), "text")
    return content, lang


def display_file(name, content, lang):
    """Show a syntax-highlighted preview of an attached file."""
    console.print(Panel(
        Syntax(content, lang, theme="monokai", line_numbers=True),
        title=f"[bold yellow]Attached: {name}[/bold yellow]",
        border_style="yellow",
        padding=(0, 1),
    ))


def pick_file():
    """Open a native file explorer dialog and return the selected path, or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(title="Select a file to attach")
    root.destroy()
    return file_path or None


def attach_file(file_path):
    """Try to attach a file. Returns (file_block, name) or prints an error and returns (None, None)."""
    content, result = read_file(file_path)
    if content is None:
        console.print(f"[bold red]{result}[/bold red]")
        return None, None
    name = Path(file_path).name
    display_file(name, content, result)
    block = f"### {name}\n```{result}\n{content}\n```"
    return block, name


def parse_input(raw):
    """Parse user input, extracting /file commands and returning (message, files_context, file_names).

    Supports:
        /file                             — open file explorer
        /file path/to/file.py             — attach a file
        /file path/to/file.py message     — attach and ask about it in one line
        Multiple /file commands on separate lines
    """
    lines = raw.strip().splitlines()
    file_blocks = []
    file_names = []
    message_parts = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "/file":
            file_path = pick_file()
            if file_path:
                block, name = attach_file(file_path)
                if block:
                    file_blocks.append(block)
                    file_names.append(name)
        elif stripped.lower().startswith("/file "):
            rest = stripped[6:].strip()
            parts = rest.split(maxsplit=1)
            file_path = parts[0]
            inline_msg = parts[1] if len(parts) > 1 else None

            block, name = attach_file(file_path)
            if block:
                file_blocks.append(block)
                file_names.append(name)
            if inline_msg:
                message_parts.append(inline_msg)
        else:
            message_parts.append(line)

    files_context = "\n\n".join(file_blocks)
    message = "\n".join(message_parts).strip()
    return message, files_context, file_names


def build_user_content(message, files_context, rag_context="", mcp_context=""):
    """Combine file context, RAG context, MCP context, and user message into a single content string."""
    parts = []
    if message:
        parts.append(message)
    if rag_context:
        parts.append(f"Use the following retrieved data to help answer:\n\n{rag_context}")
    if mcp_context:
        parts.append(f"Use the following tool results to help answer:\n\n{mcp_context}")
    if files_context:
        parts.append(f"Here are the referenced files:\n\n{files_context}")
    return "\n\n".join(parts) if parts else message
