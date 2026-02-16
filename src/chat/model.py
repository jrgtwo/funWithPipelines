"""Model loading, quantization, and context management."""

import torch
from transformers import pipeline, BitsAndBytesConfig

from .ui import console


def load_model(model_path, args):
    """Load a model pipeline and return (chatbot, max_tokens, max_context)."""
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
