"""CLI image captioning app using Hugging Face's pipeline API."""

import argparse
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from transformers import pipeline, AutoImageProcessor
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from PIL import Image

console = Console()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
MODELS_DIR = Path("./models")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions for images")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda, cpu, mps)")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max new tokens per caption")
    return parser.parse_args()


def is_vision_model(model_dir):
    """Check if a model directory contains an image processor config."""
    return (model_dir / "preprocessor_config.json").exists()


def pick_model():
    """Show an interactive list of vision-capable models from the models directory."""
    from InquirerPy import inquirer
    models = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir() and is_vision_model(p))
    if not models:
        console.print(f"[bold red]No vision models found in {MODELS_DIR.resolve()}[/bold red]")
        console.print("[dim]Image captioning requires a model with a preprocessor_config.json[/dim]")
        raise SystemExit(1)
    choice = inquirer.select(message="Select a model:", choices=models).execute()
    return str(MODELS_DIR / choice)


def pick_images():
    """Open a native file explorer dialog to select images."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    console.print("[dim]Opening file picker...[/dim]")
    filetypes = [("Image files", " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS))]
    paths = filedialog.askopenfilenames(title="Select images to caption", filetypes=filetypes, parent=root)

    root.destroy()

    if not paths:
        console.print("[bold red]No images selected.[/bold red]")
        raise SystemExit(1)

    return [Path(p) for p in paths]


def caption_image(captioner, image_path, max_tokens):
    """Run the captioning pipeline on a single image."""
    image = Image.open(image_path).convert("RGB")
    result = captioner(image, max_new_tokens=max_tokens)
    return result[0]["generated_text"]


def main():
    args = parse_args()

    console.print(Rule("[bold magenta]Image Captioner[/bold magenta]"))

    model_path = pick_model()
    selected = pick_images()

    console.print(f"\nLoading [bold cyan]{model_path}[/bold cyan]...")
    image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
    captioner = pipeline("image-to-text", model=model_path, device_map=args.device, image_processor=image_processor)
    console.print(f"[green]Model loaded.[/green] Processing {len(selected)} image(s)...\n")

    for path in selected:
        with console.status(f"[dim]Captioning {path.name}...[/dim]"):
            caption = caption_image(captioner, path, args.max_tokens)

        console.print(Panel(
            caption,
            title=f"[bold yellow]{path.name}[/bold yellow]",
            border_style="green",
            padding=(1, 2),
        ))
        console.print()

    console.print(Rule("[bold magenta]Done[/bold magenta]"))


if __name__ == "__main__":
    main()
