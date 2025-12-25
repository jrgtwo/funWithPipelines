from transformers import pipeline
import torch
import sys
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--prompt',)
# args = parser.parse_args()

# if not args.prompt:
#     exit("No prompt provided. Use --prompt to provide input text.")

print("================================")
print("Model loaded Start.")
print("================================")

pipeline = pipeline(task="text-generation", model="models/Qwen3-0.6B")

print("================================")
print("Model loaded successfully.")
print("================================")


user_prompt = input("Enter your prompt: ")
# Output:
print("================================")
print("Text Generation Start.")
print("================================")
generatedText = pipeline(user_prompt, max_new_tokens=50)
print("================================")
print("Text Generated successfully.")
print("================================")

print(generatedText[0]['generated_text'])  