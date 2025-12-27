from transformers import pipeline
from InquirerPy import prompt
import torch

from utils.get_model_paths import getModel_paths
from user_input.select_model import selectModel
from main import main

def setModel():
     model_paths = getModel_paths()
     model_path = selectModel(model_paths)
     return model_path

# model_paths = getModel_paths()
# model_path = selectModel(model_paths)

global _model_path
def startApp(resetModel=True, user_selected_task=None, log=[]):
    global _model_path
    if (resetModel == True):

        _model_path = setModel()

    try:
        shouldContinue, user_selected_task, log = main(pipeline, _model_path, user_selected_task, log)
        if shouldContinue:
            startApp(resetModel=False, user_selected_task=user_selected_task, log=log)
        else:
            startApp(resetModel=True)
    except Exception as e:
        print("An error occurred while running the main function:")
        print(e)

startApp()