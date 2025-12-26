from transformers import pipeline
from InquirerPy import prompt
import click
import torch
import sys
import argparse
import pathlib
from enter_prompt import getUserPrompt
from utils.get_model_paths import getModel_paths
from user_input.select_model import selectModel
from main import main

model_paths = getModel_paths()
model_path = selectModel(model_paths)
pipeline = pipeline(task="text-generation", model=model_path)

main(pipeline)