import torch
from utils.divider import divider
from rich.text import Text
from rich import print as rprint
from rich.align import Align
from rich.panel import Panel
from rich.markdown import Markdown
from user_input.enter_prompt import getUserPrompt
from rich.console import Console
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig

def textGenerationTask(pipeline, model_path, user_selected_task, new_user_selected_task, log):

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        quantization_config=quantization_config
    )
    
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, config=config, padding_side="left")
    
    # Fix mistral regex pattern for certain tokenizers
    if hasattr(tokenizer, 'fix_mistral_regex'):
        tokenizer.fix_mistral_regex = True
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test with a simple prompt
    test_prompt = "A list of colors: red, blue"
    model_inputs = tokenizer(test_prompt, return_tensors="pt", padding=False).to(model.device)
    
    content = model.generate(
        **model_inputs,
    )
    decoded_content = tokenizer.batch_decode(content, skip_special_tokens=True)[0]
    print("+++++++++++++++++++++++++++++++++")
    print(decoded_content)
    
    return decoded_content
  
