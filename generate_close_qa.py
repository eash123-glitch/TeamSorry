import argparse
from ast import literal_eval
import os
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from tqdm import tqdm
import numpy as np
import random
import json

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

