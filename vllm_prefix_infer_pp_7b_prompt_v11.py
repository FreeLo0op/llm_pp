# chinese prosody prediction
import re
import os
import sys
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, AutoPeftModelForCausalLM
# from vllm import LLM, SamplingParams

mode_path = '/mnt/cfs/NLP/hub_models/Qwen1.5-4B-Chat'
lora_path = '/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/pp_qwen1half_4b_chat_prompt_v19/qwen1half-4b-chat/v0-20240611-081245/checkpoint-5160/'
merged_model_path = "/mnt/cfs/SPEECH/liuhuang1/workspace/llm_pp/merged_v19_4b_ckt5160"

config = LoraConfig.from_pretrained(lora_path)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)
# load model
model = AutoModelForCausalLM.from_pretrained(mode_path,device_map="auto", torch_dtype = torch.float32)
# load lora weights
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

# merge model
merged_model = model.merge_and_unload()
merged_model.save_pretrained(merged_model_path, max_shard_size="2048MB", safe_serialization=True)

# 执行完成后，还需要将相应的token配置文件copy到vllm模型路径下
options = "cp " + mode_path + "/tokenizer* " + merged_model_path
os.system(options)