import torch
from transformers import AutoModel, AutoTokenizer

from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-72B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
).eval()
breakpoint()