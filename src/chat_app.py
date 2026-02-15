"""Simple CLI chat app using Hugging Face's pipeline API."""

import argparse
import torch
from datetime import datetime
from pathlib import Path
from transformers import pipeline, BitsAndBytesConfig
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

console = Console()

SUFFIX_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "jsx",
    ".tsx": "tsx", ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".sh": "bash", ".sql": "sql", ".html": "html", ".css": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".xml": "xml",
    ".toml": "toml", ".md": "markdown",
}


MODELS_DIR = Path("./models")
CHATS_DIR = Path("./chats")


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a local Hugging Face model")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens per response (default: model max)")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda, cpu, mps)")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit",
                        help="Quantization level (default: 4bit)")
    return parser.parse_args()


def pick_model():
    models = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())
    if not models:
        console.print(f"[bold red]No models found in {MODELS_DIR.resolve()}[/bold red]")
        raise SystemExit(1)
    from InquirerPy import inquirer
    choice = inquirer.select(message="Select a model:", choices=models).execute()
    return str(MODELS_DIR / choice)


def pick_persona():
    from InquirerPy import inquirer
    from tasks.chat.personalities import chat_personalities
    personas = chat_personalities()
    choice = inquirer.select(message="Select a persona:", choices=list(personas.keys())).execute()
    return personas[choice]


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
    preview = content
    console.print(Panel(
        Syntax(preview, lang, theme="monokai", line_numbers=True),
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
            # Split into path and optional inline message
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


def build_user_content(message, files_context):
    """Combine file context and user message into a single user content string."""
    if files_context and message:
        return f"{message}\n\nHere are the referenced files:\n\n{files_context}"
    if files_context:
        return f"Here are the referenced files:\n\n{files_context}"
    return message


def save_chat(messages):
    """Save conversation messages to a timestamped file in the chats directory."""
    CHATS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = CHATS_DIR / f"chat_{timestamp}.md"
    lines = []
    for msg in messages:
        role = msg["role"].capitalize()
        if role == "System":
            lines.append(f"**System prompt:** {msg['content']}\n")
        elif role == "User":
            lines.append(f"**You:** {msg['content']}\n")
        else:
            lines.append(f"**Assistant:** {msg['content']}\n")
    filepath.write_text("\n---\n\n".join(lines), encoding="utf-8")
    return filepath


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
        "quit                  — exit",
        title="[bold cyan]Commands[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))


def load_model(model_path, args):
    """Load a model pipeline and return (chatbot, max_tokens)."""
    torch.cuda.empty_cache()
    console.print(f"\nLoading [bold cyan]{model_path}[/bold cyan] ({args.quantize})...")

    model_kwargs = {}
    try:
        import flash_attn  # noqa: F401
        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        console.print("[dim]flash-attn not installed — using default attention.[/dim]")
    if args.quantize == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quantize == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    chatbot = pipeline(
        "text-generation",
        model=model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        model_kwargs=model_kwargs,
    )
    chatbot.tokenizer.pad_token_id = chatbot.model.config.eos_token_id
    chatbot.model.generation_config.max_length = None
    chatbot.model = torch.compile(chatbot.model, mode="reduce-overhead")
    max_tokens = args.max_tokens or getattr(chatbot.model.config, "max_position_embeddings", 2048)
    console.print(f"[dim]Max new tokens per response: {max_tokens}[/dim]")
    if chatbot.tokenizer.chat_template is None:
        chatbot.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'].upper() + ':\n' + message['content'] + '\n\n' }}"
            "{% endfor %}"
            "{{ 'ASSISTANT:\n' }}"
        )
        console.print("[dim]No chat template found — using basic fallback.[/dim]")
    max_context = getattr(chatbot.model.config, "max_position_embeddings", 2048)
    console.print("[green]Model loaded.[/green] Type [bold]/help[/bold] for commands.\n")
    return chatbot, max_tokens, max_context


def trim_messages(messages, tokenizer, max_context, max_tokens):
    """Drop oldest non-system messages so the prompt fits within the context window."""
    while len(tokenizer.apply_chat_template(messages)) > max_context - max_tokens:
        if len(messages) > 2:
            messages.pop(1)
        else:
            break


def main():
    args = parse_args()

    console.print(Rule("[bold magenta]CLI Chat[/bold magenta]"))
    model_path = pick_model()
    persona = pick_persona()
    chatbot, max_tokens, max_context = load_model(model_path, args)

    system_msg = {"role": "system", "content": persona}
    messages = [system_msg]
    attached_files = []
    pending_file_blocks = []

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break

        stripped = user_input.strip().lower()
        if stripped in ("quit", "exit"):
            break
        if stripped == "/help":
            print_help()
            continue
        if stripped == "/model":
            del chatbot
            model_path = pick_model()
            chatbot, max_tokens, max_context = load_model(model_path, args)
            messages = [system_msg]
            attached_files.clear()
            pending_file_blocks.clear()
            continue
        if stripped == "/clear":
            messages = [system_msg]
            attached_files.clear()
            pending_file_blocks.clear()
            console.print("[dim]Conversation cleared.[/dim]\n")
            continue
        if stripped == "/save":
            if len(messages) > 1:
                filepath = save_chat(messages)
                console.print(f"[green]Conversation saved to {filepath}[/green]\n")
            else:
                console.print("[dim]Nothing to save yet.[/dim]\n")
            continue
        if stripped == "/files":
            if attached_files:
                console.print("[bold yellow]Attached files:[/bold yellow] " + ", ".join(attached_files))
            else:
                console.print("[dim]No files attached yet.[/dim]")
            console.print()
            continue

        message, files_context, file_names = parse_input(user_input)
        if not message and not files_context:
            continue

        # Track attached file names for /files command
        for name in file_names:
            if name not in attached_files:
                attached_files.append(name)

        # Accumulate file blocks; only send to model when there's a message
        if files_context:
            pending_file_blocks.append(files_context)
        if not message:
            console.print("[dim]File attached. Type a message to send, or attach more files.[/dim]\n")
            continue

        all_files_context = "\n\n".join(pending_file_blocks)
        pending_file_blocks.clear()

        content = build_user_content(message, all_files_context)
        messages.append({"role": "user", "content": content})

        trim_messages(messages, chatbot.tokenizer, max_context, max_tokens)

        with console.status("[dim]Thinking...[/dim]"):
            output = chatbot(messages, max_new_tokens=max_tokens, max_length=None)

        reply = output[0]["generated_text"][-1]["content"]
        # Clean BPE token artifacts that some tokenizers leave in decoded text
        reply = reply.replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t")
        messages.append({"role": "assistant", "content": reply})

        console.print(Panel(
            Markdown(reply, code_theme="monokai"),
            title="[bold green]Assistant[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()

    if len(messages) > 1:
        filepath = save_chat(messages)
        console.print(f"[green]Conversation saved to {filepath}[/green]")
    console.print(Rule("[bold magenta]Goodbye![/bold magenta]"))


if __name__ == "__main__":
    main()
