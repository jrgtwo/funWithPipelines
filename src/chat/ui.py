"""Console setup and UI helpers."""

from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.key_binding import KeyBindings
console = Console()

_bindings = KeyBindings()


@_bindings.add("escape", "enter")  # Escape then Enter for newline
def _newline(event):
    event.current_buffer.insert_text("\n")


def get_input():
    """Prompt where Enter sends and Escape+Enter adds a newline."""
    return pt_prompt("You: ", key_bindings=_bindings)


def print_help():
    console.print(Panel(
        "/file                 — browse and attach a file\n"
        "/file <path>          — attach a file by path\n"
        "/file <path> <msg>    — attach a file and ask about it\n"
        "/files                — list currently attached files\n"
        "/dataset              — browse and load a CSV/JSON dataset for RAG\n"
        "/dataset <path>       — load a dataset by path\n"
        "/dataset clear        — unload the current dataset\n"
        "/dataset info         — show loaded dataset info\n"
        "/mcp                  — show connected MCP servers and tools\n"
        "/mcp connect <n> <cmd> [args] — connect to an MCP server via stdio\n"
        "/mcp disconnect <name> — disconnect an MCP server\n"
        "/mcp list             — list all tools across connected servers\n"
        "/mcp call <s>.<tool> [json] — call a tool, result injected into next message\n"
        "/mcp fetch <uri>      — fetch an MCP resource, result injected into next message\n"
        "/save                 — save conversation to file\n"
        "/model                — switch to a different model\n"
        "/clear                — clear conversation history\n"
        "/help                 — show this message\n"
        "quit                  — exit\n\n"
        "[dim]Tip: Press Escape then Enter for a new line[/dim]",
        title="[bold cyan]Commands[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
