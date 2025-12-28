from transformers import pipeline
import torch

from utils.get_model_paths import getModel_paths
from user_input.select_model import selectModel
from main import main

global _model_path
def setModel():
     model_paths = getModel_paths()
     model_path = selectModel(model_paths)
     return model_path

# model_paths = getModel_paths()
# model_path = selectModel(model_paths)


def startApp(resetModel=True, user_selected_task=None, log=[], latestPipeline=None):
    global _model_path
    if (resetModel == True):

        _model_path = setModel()

    try:
        shouldContinue, user_selected_task, log, newPipeline = main(
            pipeline=latestPipeline if resetModel == False else pipeline, 
            model_path=_model_path, 
            user_selected_task=user_selected_task, 
            log=log
        )

        print(f"Should Continue: {shouldContinue}")
        if shouldContinue:
            startApp(
                resetModel=False, 
                user_selected_task=user_selected_task, 
                log=log, 
                latestPipeline=newPipeline
            )
        else:
            startApp(resetModel=True)
    except Exception as e:
        print("An error occurred while running the main function:")
        print(e)

startApp()