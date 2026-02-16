"""Console setup and UI helpers."""

from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.key_binding import KeyBindings

console = Console()

_bindings = KeyBindings()


@_bindings.add("escape", "enter")
@_bindings.add("c-j")  # Alt+Enter on some terminals
def _newline(event):
    event.current_buffer.insert_text("\n")


def get_input():
    """Multiline prompt: Enter sends, Alt+Enter / Escape+Enter adds a newline."""
    return pt_prompt("You: ", key_bindings=_bindings, multiline=True)


def print_help():
    console.print(Panel(
        "/file                 — browse and attach a file\n"
        "/file <path>          — attach a file by path\n"
        "/file <path> <msg>    — attach a file and ask about it\n"
        "/files                — list currently attached files\n"
        "/save                 — save conversation to file\n"
        "/model                — switch to a different model\n"
        "/clear                — clear conversation history\n"
        "/help                 — show this message\n"
        "quit                  — exit\n\n"
        "[dim]Tip: Press Escape+Enter for a new line[/dim]",
        title="[bold cyan]Commands[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
