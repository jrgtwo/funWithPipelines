"""Constants and CLI argument parsing."""

import argparse
from pathlib import Path

SUFFIX_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "jsx",
    ".tsx": "tsx", ".java": "java", ".c": "c", ".cpp": "cpp", ".h": "cpp",
    ".cs": "csharp", ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".sh": "bash", ".sql": "sql", ".html": "html", ".css": "css",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".xml": "xml",
    ".toml": "toml", ".md": "markdown",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHATS_DIR = PROJECT_ROOT / "chats"
DATASETS_DIR = PROJECT_ROOT / "datasets"


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a local Hugging Face model")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens per response (default: 1024)")
    parser.add_argument("--device", default="cuda", help="Device map strategy (cuda, auto, cpu)")
    parser.add_argument("--quantize", choices=["4bit", "8bit", "none"], default="4bit",
                        help="Quantization level (default: 4bit)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for RAG embeddings (default: all-MiniLM-L6-v2)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of RAG chunks to retrieve per query (default: 5)")
    return parser.parse_args()
