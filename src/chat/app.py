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
from .rag import RAGIndex, pick_dataset, format_rag_context
from .mcp_client import MCPClientManager, format_mcp_context, format_resource_context


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


def _handle_mcp_command(raw: str, manager: MCPClientManager, pending_mcp_blocks: list) -> None:
    """Dispatch /mcp subcommands."""
    import json as _json

    parts = raw.split()

    if len(parts) == 1:
        # /mcp — show status
        servers = manager.list_servers()
        if not servers:
            console.print("[dim]No MCP servers connected. Use /mcp connect <name> <command> [args...][/dim]\n")
            return
        console.print(f"[bold yellow]Connected MCP servers:[/bold yellow] {', '.join(servers)}")
        try:
            all_tools = manager.list_tools()
            for server_name, tools in all_tools.items():
                console.print(f"\n  [bold]{server_name}[/bold]")
                for t in tools:
                    if "error" in t:
                        console.print(f"    [red]Error: {t['error']}[/red]")
                    else:
                        console.print(f"    [cyan]{t['name']}[/cyan] — {t['description']}")
        except Exception as e:
            console.print(f"[bold red]Error listing tools: {e}[/bold red]")
        console.print()
        return

    sub = parts[1].lower()

    if sub == "connect":
        # /mcp connect <name> <command> [args...]
        if len(parts) < 4:
            console.print("[bold red]Usage: /mcp connect <name> <command> [args...][/bold red]\n")
            return
        name = parts[2]
        command = parts[3]
        args = parts[4:]
        with console.status(f"[dim]Connecting to '{name}'...[/dim]", spinner="dots"):
            try:
                manager.connect(name, command, args)
            except Exception as e:
                console.print(f"[bold red]Failed to connect: {e}[/bold red]\n")
                return
        console.print(f"[green]Connected to '{name}' ({command})[/green]\n")
        return

    if sub == "disconnect":
        if len(parts) < 3:
            console.print("[bold red]Usage: /mcp disconnect <name>[/bold red]\n")
            return
        name = parts[2]
        try:
            manager.disconnect(name)
            console.print(f"[dim]Disconnected '{name}'.[/dim]\n")
        except KeyError as e:
            console.print(f"[bold red]{e}[/bold red]\n")
        return

    if sub == "list":
        try:
            all_tools = manager.list_tools()
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]\n")
            return
        if not all_tools:
            console.print("[dim]No MCP servers connected.[/dim]\n")
            return
        for server_name, tools in all_tools.items():
            console.print(f"\n[bold yellow]{server_name}[/bold yellow]")
            for t in tools:
                if "error" in t:
                    console.print(f"  [red]{t['error']}[/red]")
                else:
                    console.print(f"  [cyan]{t['name']}[/cyan] — {t['description']}")
        console.print()
        return

    if sub == "call":
        # /mcp call <server>.<tool> [json_args]
        if len(parts) < 3:
            console.print("[bold red]Usage: /mcp call <server>.<tool> [json_args][/bold red]\n")
            return
        dotted = parts[2]
        if "." not in dotted:
            console.print("[bold red]Specify tool as <server>.<tool>, e.g.: /mcp call myserver.generate[/bold red]\n")
            return
        server_name, tool_name = dotted.split(".", 1)
        prefix = "/mcp call " + dotted
        raw_json = raw[len(prefix):].strip()
        if raw_json:
            try:
                arguments = _json.loads(raw_json)
            except _json.JSONDecodeError as e:
                console.print(f"[bold red]Invalid JSON arguments: {e}[/bold red]\n")
                return
        else:
            arguments = {}
        with console.status(f"[dim]Calling {server_name}.{tool_name}...[/dim]", spinner="dots"):
            try:
                result = manager.call_tool(server_name, tool_name, arguments)
            except Exception as e:
                console.print(f"[bold red]Tool call failed: {e}[/bold red]\n")
                return
        block = format_mcp_context(server_name, tool_name, result)
        pending_mcp_blocks.append(block)
        console.print(f"[green]Tool result from {server_name}.{tool_name} queued for next message.[/green]")
        console.print(f"[dim]{result[:200]}{'...' if len(result) > 200 else ''}[/dim]\n")
        return

    if sub == "fetch":
        # /mcp fetch <uri>
        if len(parts) < 3:
            console.print("[bold red]Usage: /mcp fetch <uri>[/bold red]\n")
            return
        uri = parts[2]
        with console.status(f"[dim]Fetching {uri}...[/dim]", spinner="dots"):
            try:
                result = manager.fetch_resource(uri)
            except Exception as e:
                console.print(f"[bold red]Resource fetch failed: {e}[/bold red]\n")
                return
        block = format_resource_context(uri, result)
        pending_mcp_blocks.append(block)
        console.print(f"[green]Resource {uri} queued for next message.[/green]")
        console.print(f"[dim]{result[:200]}{'...' if len(result) > 200 else ''}[/dim]\n")
        return

    console.print(f"[bold red]Unknown /mcp subcommand: '{sub}'[/bold red]")
    console.print("[dim]Use /help for available commands.[/dim]\n")


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
    rag_index = RAGIndex()
    rag_active = False
    mcp_manager = MCPClientManager()
    pending_mcp_blocks: list = []

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
            pending_mcp_blocks.clear()
            rag_index = RAGIndex()
            rag_active = False
            continue
        if stripped == "/clear":
            messages = [system_msg]
            attached_files.clear()
            pending_file_blocks.clear()
            pending_mcp_blocks.clear()
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
        if stripped == "/dataset":
            file_path = pick_dataset()
            if file_path:
                try:
                    if rag_index.load_cache(file_path):
                        count = len(rag_index.chunks)
                        console.print(f"[green]Loaded cached index for {rag_index.source_name} ({count} chunks)[/green]\n")
                    else:
                        count = rag_index.build(file_path, args.embedding_model)
                        console.print(f"[green]Indexed {count} chunks from {rag_index.source_name}[/green]\n")
                    rag_active = True
                except Exception as e:
                    console.print(f"[bold red]Error loading dataset: {e}[/bold red]\n")
            continue
        if stripped.startswith("/dataset "):
            rest = user_input.strip()[9:].strip()
            if rest.lower() == "clear":
                rag_index = RAGIndex()
                rag_active = False
                console.print("[dim]Dataset cleared.[/dim]\n")
                continue
            if rest.lower() == "info":
                if rag_active:
                    console.print(f"[bold yellow]Dataset:[/bold yellow] {rag_index.source_name} ({len(rag_index.chunks)} chunks)")
                else:
                    console.print("[dim]No dataset loaded.[/dim]")
                console.print()
                continue
            file_path = rest
            try:
                if rag_index.load_cache(file_path):
                    count = len(rag_index.chunks)
                    console.print(f"[green]Loaded cached index for {rag_index.source_name} ({count} chunks)[/green]\n")
                else:
                    count = rag_index.build(file_path, args.embedding_model)
                    console.print(f"[green]Indexed {count} chunks from {rag_index.source_name}[/green]\n")
                rag_active = True
            except Exception as e:
                console.print(f"[bold red]Error loading dataset: {e}[/bold red]\n")
            continue

        if stripped == "/mcp" or stripped.startswith("/mcp "):
            _handle_mcp_command(user_input.strip(), mcp_manager, pending_mcp_blocks)
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

        all_mcp_context = "\n\n".join(pending_mcp_blocks)
        pending_mcp_blocks.clear()

        rag_context = ""
        if rag_active and message:
            with console.status("[dim]Searching dataset...[/dim]", spinner="dots"):
                rag_chunks = rag_index.query(message, top_k=args.top_k, embedding_model_name=args.embedding_model)
                rag_context = format_rag_context(rag_chunks, rag_index.source_name)

        content = build_user_content(message, all_files_context, rag_context, all_mcp_context)
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
            stop_strings=["USER:", "SYSTEM:"],
            tokenizer=tokenizer,
            streamer=streamer,
        )

        def _generate():
            with torch.inference_mode():
                model.generate(**gen_kwargs)

        thread = threading.Thread(target=_generate)

        console.print(Rule("[bold green]Assistant[/bold green]", style="green"))
        thread.start()
        chunks = []
        with console.status("[dim]Generating...[/dim]", spinner="dots"):
            for text in streamer:
                chunks.append(text)
        thread.join()
        del input_ids, attention_mask
        torch.cuda.empty_cache()

        reply = "".join(chunks)
        reply = reply.replace("\u0120", " ").replace("\u010a", "\n").replace("\u0109", "\t")
        console.print(Markdown(reply))
        messages.append({"role": "assistant", "content": reply})
        console.print()

    if len(messages) > 1:
        filepath = save_chat(messages)
        console.print(f"[green]Conversation saved to {filepath}[/green]")
    console.print(Rule("[bold magenta]Goodbye![/bold magenta]"))


if __name__ == "__main__":
    main()
