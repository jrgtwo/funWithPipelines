"""Main chat loop and entry point."""

import threading

import torch
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from transformers import TextIteratorStreamer

from .config import MODELS_DIR, parse_args
from .ui import console, print_help, get_input
from .model import load_model, trim_messages
from .files import parse_input, build_user_content
from .history import save_chat


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
    from .personalities import chat_personalities
    personas = chat_personalities()
    choice = inquirer.select(message="Select a persona:", choices=list(personas.keys())).execute()
    return personas[choice]


def main():
    args = parse_args()

    console.print(Rule("[bold magenta]CLI Chat[/bold magenta]"))
    model_path = pick_model()
    persona = pick_persona()
    model, tokenizer, max_tokens, max_context = load_model(model_path, args)

    system_msg = {"role": "system", "content": persona}
    messages = [system_msg]
    attached_files = []
    pending_file_blocks = []

    while True:
        try:
            user_input = get_input()
        except (KeyboardInterrupt, EOFError):
            break

        stripped = user_input.strip().lower()
        if stripped in ("quit", "exit"):
            break
        if stripped == "/help":
            print_help()
            continue
        if stripped == "/model":
            del model
            model_path = pick_model()
            model, tokenizer, max_tokens, max_context = load_model(model_path, args)
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

        console.print(Rule("[bold blue]You[/bold blue]", style="blue"))
        console.print(user_input.strip())
        console.print()

        message, files_context, file_names = parse_input(user_input)
        if not message and not files_context:
            continue

        for name in file_names:
            if name not in attached_files:
                attached_files.append(name)

        if files_context:
            pending_file_blocks.append(files_context)
        if not message:
            console.print("[dim]File attached. Type a message to send, or attach more files.[/dim]\n")
            continue

        all_files_context = "\n\n".join(pending_file_blocks)
        pending_file_blocks.clear()

        content = build_user_content(message, all_files_context)
        messages.append({"role": "user", "content": content})

        trim_messages(messages, tokenizer, max_context, max_tokens)

        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

        def _generate():
            with torch.inference_mode():
                model.generate(**gen_kwargs)

        thread = threading.Thread(target=_generate)

        console.print(Rule("[bold green]Assistant[/bold green]", style="green"))
        thread.start()
        chunks = []
        for text in streamer:
            console.print(text, end="")
            chunks.append(text)
        thread.join()
        del input_ids, attention_mask
        torch.cuda.empty_cache()

        reply = "".join(chunks)
        reply = reply.replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t")
        messages.append({"role": "assistant", "content": reply})
        console.print("\n")

    if len(messages) > 1:
        filepath = save_chat(messages)
        console.print(f"[green]Conversation saved to {filepath}[/green]")
    console.print(Rule("[bold magenta]Goodbye![/bold magenta]"))


if __name__ == "__main__":
    main()
