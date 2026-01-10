
from transformers import pipeline
from main import main

from user_input.model import selectModel, setModel

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