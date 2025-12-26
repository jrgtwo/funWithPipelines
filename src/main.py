from enter_prompt import getUserPrompt
from user_input.select_task import selectTask
import torch

user_task = "text-generation"
model_path_set = False

def main(pipeline, model_path):
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {device}")
    print("============================")
    print(" ")
    user_selected_task = selectTask()

    newPipeline = pipeline(task=user_selected_task, model=model_path,  device=device)

    print(" ")
    print("============================")
    print(" ")


    user_prompt = getUserPrompt(pipeline)

    # Output:
    print("============================")
    print("  Text Generation Started")
    generatedText = newPipeline(user_prompt, max_new_tokens=250)
    print("============================")
    print("  Text Generation Complete")
    print("============================")
    print(" ")

    print(generatedText[0])  

    print(" ")
    wait = input("Press Enter to continue...")
    print(" ")

    print(pipeline)
    print(model_path)



    main(pipeline, model_path)