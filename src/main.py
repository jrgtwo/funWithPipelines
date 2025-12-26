from enter_prompt import getUserPrompt
from user_input.select_task import selectTask

user_task = "text-generation"
model_path_set = False

def main(pipeline, model_path):
    
    print("============================")
    print(" ")
    user_selected_task = selectTask()

    newPipeline = pipeline(task=user_selected_task, model=model_path)

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