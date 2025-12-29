
from InquirerPy import prompt
from utils.get_model_paths import getModel_paths

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

def setModel():
     model_paths = getModel_paths()
     model_path = selectModel(model_paths)
     return model_path