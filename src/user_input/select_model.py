from InquirerPy import prompt

def selectModel(model_paths):
    questions = [
        {
            "type": "list",
            "name":"model_path",
            "message": "Which model do you want to use?",
            "choices": model_paths,
        }
    ]
    result = prompt(questions)
    model_path = result["model_path"]

    return model_path