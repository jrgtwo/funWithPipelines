
def getUserPrompt():
    user_prompt = input("Enter your prompt: ")
    print('\033[1A' + '\033[K', end='')

    return user_prompt
