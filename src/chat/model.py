"""Model loading, quantization, and context management."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .ui import console


def load_model(model_path, args):
    """Load model and tokenizer, return (model, tokenizer, max_tokens, max_context)."""
    torch.cuda.empty_cache()
    console.print(f"\nLoading [bold cyan]{model_path}[/bold cyan] ({args.quantize})...")

    model_kwargs = {"attn_implementation": "sdpa"}
    if args.quantize == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.quantize == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **model_kwargs,
    )

    # Warn if layers were offloaded to CPU (causes severe slowdowns)
    if hasattr(model, "hf_device_map"):
        offloaded = [k for k, v in model.hf_device_map.items() if v == "cpu"]
        if offloaded:
            console.print(
                f"[bold yellow]Warning:[/bold yellow] {len(offloaded)} layer(s) offloaded to CPU. "
                "Inference will be very slow. Use a smaller model or --quantize 4bit."
            )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = model.config.eos_token_id
    model.generation_config.max_length = None
    model.eval()
    max_tokens = args.max_tokens or min(getattr(model.config, "max_position_embeddings", 2048), 1024)
    console.print(f"[dim]Max new tokens per response: {max_tokens}[/dim]")
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'].upper() + ':\n' + message['content'] + eos_token + '\n\n' }}"
            "{% endfor %}"
            "{{ 'ASSISTANT:\n' }}"
        )
        console.print("[dim]No chat template found — using basic fallback.[/dim]")
    max_context = getattr(model.config, "max_position_embeddings", 2048)
    console.print("[green]Model loaded.[/green] Type [bold]/help[/bold] for commands.\n")
    return model, tokenizer, max_tokens, max_context


def trim_messages(messages, tokenizer, max_context, max_tokens):
    """Drop oldest non-system messages so the prompt fits within the context window."""
    while len(tokenizer.apply_chat_template(messages)) > max_context - max_tokens:
        if len(messages) > 2:
            messages.pop(1)
        else:
            break
