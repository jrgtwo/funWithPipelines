from enter_prompt import getUserPrompt

def main(pipeline):
    print("============================")
    print(" ")
    user_prompt = getUserPrompt(pipeline)

    # Output:
    print("============================")
    print("  Text Generation Started")
    generatedText = pipeline(user_prompt, max_new_tokens=250)
    print("============================")
    print("  Text Generation Complete")
    print("============================")
    print(" ")

    print(generatedText[0]['generated_text'])  

    print(" ")
    wait = input("Press Enter to continue...")
    print(" ")



    main(pipeline)