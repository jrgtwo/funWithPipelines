from enter_prompt import getUserPrompt
from user_input.select_task import selectTask
import torch
import accelerate

from utils.divider import divider

def main(pipeline, model_path, log):
    torch.cuda.empty_cache()

    divider()

    user_selected_task = selectTask()

    newPipeline = pipeline(
        task=user_selected_task, 
        model=model_path,
        tokenizer=(model_path), 
        dtype=(torch.bfloat16),
        batch_size=2,
        device_map="auto"
    )

    divider()

    user_prompt = getUserPrompt(pipeline)

    # Output:
    divider("Text Generation Started")
    generatedText = newPipeline(user_prompt, max_new_tokens=250)
    divider("Text Generation Complete")

    print(generatedText[0]["generated_text"])  

    divider()
    wait = input("Press Enter to continue...")
    divider()
    shouldContinue = True
    # main(pipeline, model_path)
    return shouldContinue, generatedText