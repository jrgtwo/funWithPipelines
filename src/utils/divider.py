from rich import print
from rich.text import Text
from rich.panel import Panel

def divider(textToPrint=""):
    print(" ")
    
    text = Text()
    text.append(textToPrint)
    text.stylize("bold gray")
    text.justify = "left"
    panel = Panel(text, expand=True, border_style="magenta")
    
    print(panel)
    print(" ")
    