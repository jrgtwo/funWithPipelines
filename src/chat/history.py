"""Conversation saving and persistence."""

from datetime import datetime

from .config import CHATS_DIR


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
