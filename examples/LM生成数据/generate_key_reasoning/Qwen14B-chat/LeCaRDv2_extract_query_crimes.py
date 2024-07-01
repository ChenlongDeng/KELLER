from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--rank', default=0, type=int)
args = parser.parse_args()

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-chat", device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True).eval()

# label
label_top30 = json.load(open('/share/kelong/chenlong/Legal/data/LeCaRD/data/label/label_top30_dict.json', 'r'))

# 读取querys
query_path = '/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/query.json'
querys = {}
with open(query_path, 'r') as f:
    for line in f:
        query = json.loads(line)
        querys[str(query['id'])] = query['query'].split('：', 1)[0]
output_path = f'./LeCaRDv2_query_crimes.json'
if os.path.exists(output_path):
    output_data = json.load(open(output_path, 'r'))
else:
    output_data = {}


template = """请基于给定的法律文书标题，生成该标题中包含的罪名（可能有一个或多个，多个时用分号隔开）。请学习以下例子的回复格式，直接生成罪名。
标题：长春放火、非法持有、私藏枪支、弹药一审刑事判决书山东省沂南县人民法院刑事判决书（2018）鲁1321刑初112号
包含罪名：放火罪；非法持有、私藏枪支、弹药罪

标题：䎋某、卜某某引诱、容留、介绍卖淫一审刑事判决书山东省青岛市黄岛区人民法院刑事判决书（2019）鲁0211刑初141号
包含罪名：引诱、容留、介绍卖淫罪

标题：冯素军引诱、容留、介绍卖淫罪、拒不执行判决、裁定罪张青青、杨小兵等引诱、容留、介绍卖淫罪一审刑事判决书江苏省如东县人民法院刑事判决书（2019）苏0623刑初105号
包含罪名：引诱、容留、介绍卖淫罪；拒不执行判决、裁定罪

标题：```{{标题}}```
包含罪名："""

    
for qid in tqdm(querys.keys()):
    title = tokenizer.decode(tokenizer(querys[qid], truncation=True, max_length=100)['input_ids'], skip_special_tokens=True)
    response, history = model.chat(tokenizer, template.replace("{{标题}}", title), history=None)
    output_data[qid] = list(set(response.split('；')))
    # if len(output_data) % 10 == 0:
    with open(output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

