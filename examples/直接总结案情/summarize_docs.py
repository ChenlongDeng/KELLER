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



# query
test_querys = {}
with open('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/test_query.jsonl', 'r') as f:
    for line in f:
        query = json.loads(line)
        test_querys[str(query['id'])] = query['fact']
        
# label
labels = {}
with open('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/label/relevence.trec', 'r') as f:
    for line in f:
        qid, _, did, rel = line.strip().split('\t')
        if qid not in labels.keys():
            labels[qid] = {}
        labels[qid][did] = int(rel)
LeCaRDv2_bm25 = json.load(open('/share/kelong/chenlong/Legal/Ranking_LeCaRDv2/examples/生成bm25结果/bm25_top1k.json', 'r'))

# doc: label+bm25
doc_dir = '/share/kelong/chenlong/Legal/data/LeCaRDv2-main/candidate'
docs = {}
for filename in tqdm(os.listdir(doc_dir), ncols=120):
    doc = json.load(open(os.path.join(doc_dir, filename), 'r'))
    docs[filename.split('.')[0]] = doc
test_doc_ids = []
for qid in test_querys.keys():
    test_doc_ids.extend(list(labels[qid].keys()))
    if all([labels[qid][did] != 3 for did in labels[qid].keys()]):
        golden_labels = [2,3]
    else:
        golden_labels = [3]
    if all(labels[qid][did] in golden_labels for did in labels[qid].keys()):
        extra_candidates = [i for i in LeCaRDv2_bm25[qid].keys() if (i not in labels[qid].keys()) or (labels[qid][i] not in golden_labels)]
        for did in extra_candidates[100:110]:
            test_doc_ids.append(did)
test_doc_ids = list(set(test_doc_ids))

output_path = f'./test_docs.json'
if os.path.exists(output_path):
    output_data = json.load(open(output_path, 'r'))
else:
    output_data = {}

# Note: The default behavior now has injection attack prevention off.
model_dir = "Qwen/Qwen-72B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vllm_model = vLLMWrapper(model_dir, quantization='gptq', dtype="float16", gpu_memory_utilization=0.9, tensor_parallel_size=2)

for did in tqdm(test_doc_ids):
    original_fact = tokenizer.decode(tokenizer(docs[did]['fact'], truncation=True, max_length=2000)['input_ids'], skip_special_tokens=True)
    response, history = vllm_model.chat(original_fact, history=None, system='请在100字内直接总结以下法律案件的情节')
    output_data[did] = response

    with open(output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

