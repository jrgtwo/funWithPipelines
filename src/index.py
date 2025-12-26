from transformers import pipeline
from InquirerPy import prompt
import torch

from utils.get_model_paths import getModel_paths
from user_input.select_model import selectModel
from main import main

model_paths = getModel_paths()
model_path = selectModel(model_paths)


main(pipeline, model_path)