import os
import torch
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig
)

# Define project root and import constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '....'))
sys.path.append(PROJECT_ROOT)
from constants import TRAIN_DIRNAME, TEST_DIRNAME, VAL_DIRNAME, IMAGE_DIRNAME

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map=device
)

def create_prompt(question):
    """Creates a prompt for the model."""
    prompt = f""" ### INSTRUCTION:
    Your task is to answer the question based on the given image. You can only answer 'yes' or 'no'.
    ### USER: <image>
    {question}
    ### ASSISTANT:
    """
    return prompt

# Configure generation
generation_config = GenerationConfig(
    max_new_tokens=10,
    do_sample=True,
    temperature=0.1,
    top_p=0.95,
    top_k=50,
    eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,
)

def get_data(path: str):
    """Loads data from a given file path."""
    data = []
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                tmp = line.strip().split('\t')
                if len(tmp) < 2:
                    print(f"Warning: Malformed line (skipped): {line}")
                    continue
                
                qa = tmp[1].split('?')
                if len(qa) >= 2:
                    question = qa[0] + '?'
                    answer = qa[1].strip() if len(qa) == 2 else qa[2].strip()
                    data.append({
                        'image_path': tmp[0].strip()[:-2],  
                        'question': question,
                        'answer': answer
                    })
                else:
                    print(f"Warning: Skipping malformed question-answer pair: {line}")
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
    
    return data

# Load test data
test_data = get_data(TEST_DIRNAME)
idx = 0
question = test_data[idx]['question']
image_name = test_data[idx]['image_path']
image_path = os.path.join(IMAGE_DIRNAME, image_name)
label = test_data[idx]['answer']
image = Image.open(image_path)

# Generate response
prompt = create_prompt(question)
inputs = processor(prompt, image, padding=True, return_tensors="pt").to(device)
output = model.generate(**inputs, generation_config=generation_config)
generated_text = processor.decode(output[0], skip_special_tokens=True)

# Display results
plt.imshow(image)
plt.axis("off")
plt.show()
print(f"Question: {question}")
print(f"Label: {label}")
print(f"Prediction: {generated_text.split('### ASSISTANT:')[-1]}")
