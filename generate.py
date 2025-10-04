import pickle
import json
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from tqdm import tqdm

token = ''

model_id = 'meta-llama/Llama-2-13b-chat-hf'
batch_size =4

tokenizer= AutoTokenizer.from_pretrained(model_id, padding_side = 'left', token = token)

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    token=token,
    device_map="auto",
    torch_dtype=torch.float16
)

pipeline.tokenizer.pad_token = pipeline.tokenizer.eos_token
pipeline.model.config.pad_token_id = pipeline.model.config.eos_token_id




with open ('em_train_narrations.pkl', 'rb') as f:
    data = pickle.load(f)

# print(data[0])