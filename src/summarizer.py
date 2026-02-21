"""CLI text summarization app using Hugging Face's pipeline API."""

import argparse
import json
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from transformers import pipeline, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

console = Console()

MODELS_DIR = Path("./models")
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs",
    ".rb", ".php", ".sh", ".sql", ".html", ".css", ".json", ".yaml", ".yml",
    ".xml", ".toml", ".csv", ".log",
}
SUFFIX_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".go": "go", ".rs": "rust", ".rb": "ruby",
    ".php": "php", ".sh": "bash", ".sql": "sql", ".html": "html", ".css": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".xml": "xml",
    ".toml": "toml", ".md": "markdown",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize text files")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda, cpu, mps)")
    parser.add_argument("--min-length", type=int, default=30, help="Minimum summary length in tokens")
    parser.add_argument("--max-length", type=int, default=150, help="Maximum summary length in tokens")
    return parser.parse_args()


SEQ2SEQ_ARCHITECTURES = {
    "bart", "mbart", "pegasus", "t5", "mt5", "led", "longt5", "bigbird_pegasus",
    "blenderbot", "prophetnet", "flan-t5", "nllb", "plbart",
}


def is_summarization_model(model_dir):
    """Check if a model directory contains a seq2seq architecture via config.json."""
    config_file = model_dir / "config.json"
    if not config_file.exists():
        return False
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    model_type = config.get("model_type", "").lower()
    architectures = [a.lower() for a in config.get("architectures", [])]
    if model_type in SEQ2SEQ_ARCHITECTURES:
        return True
    return any("conditionalgener" in a or "seq2seq" in a for a in architectures)


def pick_model():
    """Show an interactive list of summarization-capable models."""
    from InquirerPy import inquirer
    models = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir() and is_summarization_model(p))
    if not models:
        console.print(f"[bold red]No summarization models found in {MODELS_DIR.resolve()}[/bold red]")
        console.print("[dim]Summarization requires a seq2seq model (BART, T5, Pegasus, etc.)[/dim]")
        raise SystemExit(1)
    choice = inquirer.select(message="Select a model:", choices=models).execute()
    return str(MODELS_DIR / choice)


def pick_files():
    """Open a native file explorer dialog to select text files."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    console.print("[dim]Opening file picker...[/dim]")
    filetypes = [
        ("Text files", " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)),
        ("All files", "*.*"),
    ]
    paths = filedialog.askopenfilenames(title="Select files to summarize", filetypes=filetypes, parent=root)

    root.destroy()

    if not paths:
        console.print("[bold red]No files selected.[/bold red]")
        raise SystemExit(1)

    return [Path(p) for p in paths]


def read_file(path):
    """Read a text file, return its content or None on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        console.print(f"[bold red]Cannot read binary file: {path}[/bold red]")
        return None


def display_source(name, content):
    """Show a preview of the source file."""
    lang = SUFFIX_TO_LANG.get(Path(name).suffix.lower(), "text")
    lines = content.splitlines()
    preview = "\n".join(lines[:20]) + ("\n..." if len(lines) > 20 else "")
    console.print(Panel(
        Syntax(preview, lang, theme="monokai", line_numbers=True),
        title=f"[bold yellow]{name}[/bold yellow]",
        subtitle=f"[dim]{len(lines)} lines, {len(content)} chars[/dim]",
        border_style="yellow",
        padding=(0, 1),
    ))


def main():
    args = parse_args()

    console.print(Rule("[bold magenta]Text Summarizer[/bold magenta]"))

    model_path = pick_model()
    selected = pick_files()

    console.print(f"\nLoading [bold cyan]{model_path}[/bold cyan]...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    summarizer = pipeline("summarization", model=model_path, device_map=args.device, tokenizer=tokenizer)
    summarizer.model.generation_config.max_length = None
    summarizer.model.generation_config.max_new_tokens = None
    max_input_tokens = tokenizer.model_max_length
    console.print(f"[green]Model loaded.[/green] Processing {len(selected)} file(s)...\n")

    for path in selected:
        content = read_file(path)
        if content is None:
            continue

        # Truncate input to model's max sequence length before tokenizer warns
        content = tokenizer.decode(
            tokenizer.encode(content, max_length=max_input_tokens, truncation=True),
            skip_special_tokens=True,
        )

        display_source(path.name, content)

        with console.status(f"[dim]Summarizing {path.name}...[/dim]"):
            result = summarizer(content, min_length=args.min_length, max_length=args.max_length, truncation=True)

        summary = result[0]["summary_text"]

        console.print(Panel(
            summary,
            title=f"[bold green]Summary — {path.name}[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()

    console.print(Rule("[bold magenta]Done[/bold magenta]"))


if __name__ == "__main__":
    main()
