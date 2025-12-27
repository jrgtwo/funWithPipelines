from enter_prompt import getUserPrompt
from user_input.select_task import selectTask
import torch
import accelerate

from utils.divider import divider

def main(pipeline, model_path, user_selected_task, log):
    torch.cuda.empty_cache()

    divider()

    new_user_selected_task = user_selected_task or selectTask()

    newPipeline = pipeline(
        task=new_user_selected_task, 
        model=model_path,
        tokenizer=(model_path), 
        dtype=(torch.bfloat16),
        batch_size=2,
        device_map="auto"
    )

    divider()

   
    # user_prompt = None
    # if log:
    #     user_prompt = log[0]['generated_text']+ '\n\n' + getUserPrompt(pipeline)
    # else:
    user_prompt = getUserPrompt(newPipeline)
 
    # Output:
    divider("Text Generation Started")
    
    generatedText = newPipeline(user_prompt, max_new_tokens=250, do_sample=True, top_p=0.9, temperature=0.7)
    
    divider("Text Generation Complete")

    print(generatedText[0]["generated_text"])  

    divider()
    wait = input("Press Enter to continue...")
    print(wait)
    divider()
    shouldContinue = True
    # main(pipeline, model_path)
    return shouldContinue, new_user_selected_task, generatedText