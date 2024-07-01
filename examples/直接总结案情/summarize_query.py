from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from vllm_wrapper import vLLMWrapper
from tqdm import tqdm
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--rank', default=0, type=int)
args = parser.parse_args()

# Note: The default behavior now has injection attack prevention off.
model_dir = "Qwen/Qwen-72B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vllm_model = vLLMWrapper(model_dir, quantization='gptq', dtype="float16", gpu_memory_utilization=0.9, tensor_parallel_size=2)

# label
test_querys = {}
with open('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/test_query.jsonl', 'r') as f:
    for line in f:
        query = json.loads(line)
        test_querys[str(query['id'])] = query['fact']

output_path = f'./test_querys.json'
if os.path.exists(output_path):
    output_data = json.load(open(output_path, 'r'))
else:
    output_data = {}

    
for qid in tqdm(test_querys.keys()):
    original_fact = tokenizer.decode(tokenizer(test_querys[qid], truncation=True, max_length=2000)['input_ids'], skip_special_tokens=True)
    response, history = vllm_model.chat(original_fact, history=None, system='请在100字内直接总结以下法律案件的情节')
    output_data[qid] = response

    with open(output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

