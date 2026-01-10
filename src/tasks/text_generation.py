import torch
from utils.divider import divider
from rich.text import Text
from rich import print as rprint
from rich.align import Align
from rich.panel import Panel
from rich.markdown import Markdown
from user_input.enter_prompt import getUserPrompt
from user_input.select_persona import select_persona

from rich.console import Console
from transformers import AutoTokenizer

def textGenerationTask(pipeline, model_path, user_selected_task, new_user_selected_task, log):
    console = Console()
    persona = None
    user_prompt = getUserPrompt()

    if user_selected_task:
        persona = select_persona()    
    try:
        default_system_prompt = {"role": "system", "content": persona}

        chat =  [default_system_prompt, {"role": "user", "content": user_prompt}] 
        if  user_selected_task:
            transcript = log[0]['generated_text']
            transcript.append({"role": "user", "content": user_prompt})
            chat = transcript 
    except Exception as e:
        print("An error occurred while preparing the prompt:")
        print(e)
        return False, new_user_selected_task, log, pipeline

    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    rprint(
        Align(
            Panel(
                user_prompt,
                title="Prompt", 
                border_style="blue", 
                width=80
            )
        , align="left"
        )
    )

    generatedText = pipeline(
        max_new_tokens=(1000000),
        tokenizer=tokenizer,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        text_inputs=chat,
        pad_token_id=tokenizer.eos_token_id
    )
    
    console.print(
        Align(
            Panel(
                Markdown(generatedText[-1]["generated_text"][-1]['content']),
                title="Generated Text (Markdown)",
                width=80
            )
        , align="right"
        ), 
        justify="right",
       
    )

    return generatedText