from user_input.enter_prompt import getUserPrompt
from user_input.select_task import selectTask
import torch
import accelerate
from transformers import pipeline
from rich import print as rprint
from rich.prompt import Confirm
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from utils.divider import divider
from rich.json import JSON
import json
from rich.markdown import Markdown
from rich.align import Align
from tasks.text_generation import textGenerationTask

def main(pipeline, model_path, user_selected_task, log):
    torch.cuda.empty_cache()

    if not user_selected_task:
        divider('Select a task for the model to perform')

    new_user_selected_task = user_selected_task or selectTask()

    newPipeline = None
    if not user_selected_task:
        newPipeline = pipeline(
            task=new_user_selected_task,
            model=model_path,
            dtype=(torch.bfloat16),
            device=0
        )
        divider(f"Current Task:  {Text(new_user_selected_task)}")

    else:
        newPipeline = pipeline

    match new_user_selected_task:
        case 'text-generation':
           generatedText = textGenerationTask(newPipeline, model_path, user_selected_task, new_user_selected_task, log)
        case default:
            print(f"Task '{new_user_selected_task}' is not yet implemented.")

    waitText = Text()
    waitText.append("Press Y to continue, N to exit...")
    waitText.stylize("bold green")
    shouldContinue = Confirm.ask()
    print("\033[A\033[K", end="\r")
    
    return shouldContinue, new_user_selected_task, generatedText, newPipeline