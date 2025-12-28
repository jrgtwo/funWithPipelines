from enter_prompt import getUserPrompt
from user_input.select_task import selectTask
import torch
import accelerate
from transformers import pipeline
from rich import print
from rich.prompt import Confirm
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from utils.divider import divider
from rich.json import JSON
import json

def main(pipeline, model_path, user_selected_task, log):
    torch.cuda.empty_cache()
    
    console = Console()

    if not user_selected_task:
        divider('Select a task for the model to perform')

    new_user_selected_task = user_selected_task or selectTask()

    if not user_selected_task:
        newPipeline = pipeline(
            task=new_user_selected_task,
            model=model_path,
            dtype=(torch.bfloat16),
            device_map="auto"
        )
    else:
        newPipeline = pipeline
    
    divider(f"Current Task:  {Text(new_user_selected_task)}")

    user_prompt = getUserPrompt(newPipeline)
 
    # Output:
    divider("Text Generation Started")
    try:
        default_system_prompt = {"role": "system", "content": "I am a helpful assistant who is knowledgeable about space and the cosmos."}

        chat =  [default_system_prompt, {"role": "user", "content": user_prompt}] 
        if  user_selected_task:
            transcript = log[0]['generated_text']
            transcript.append({"role": "user", "content": user_prompt})
            chat = transcript 

    except Exception as e:
        print("An error occurred while preparing the prompt:")
        print(e)
        return False, new_user_selected_task, log, newPipeline

    generatedText = newPipeline( max_new_tokens=250, do_sample=True, top_p=0.9, temperature=0.7, text_inputs=chat)
    
    divider("Text Generation Complete")
    print(Panel(json.dumps(generatedText[0]["generated_text"], indent=4), title="Generated Text", border_style="blue"))

    print(" ")
    waitText = Text()
    waitText.append("Press Y to continue, N to exit...")
    waitText.stylize("bold green")
    shouldContinue = Confirm.ask()
    print(" ")
  
    return shouldContinue, new_user_selected_task, generatedText, newPipeline