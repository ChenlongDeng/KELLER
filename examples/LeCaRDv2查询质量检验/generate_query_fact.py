import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rank', default=0, type=int)
args = parser.parse_args()

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-chat", device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True).eval()

query_path = '/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/query.json'
court_data = {}
proc_data = {}
with open(query_path, 'r') as f:
    for line in f:
        query = json.loads(line)
        if ('经审理查明' in query['query']):
            court_data[str(query['id'])] = query['query']
        else:
            proc_data[str(query['id'])] = query['query']
            
court_template = """请抽取出所提供案件文书中“经审理查明”部分的事实。请学习以下一个例子的抽取方式：
案件文书：杨杰信用卡诈骗一审刑事判决书广东省珠海市香洲区人民法院刑事判决书珠香法刑初字第748号：珠海市香洲区人民检察院以珠香检公诉刑诉起诉书指控被告人杨杰犯信用卡诈骗罪，向本院提起公诉。本院依法组成合议庭，公开开庭审理了本案。现已审理终结经审理查明，被告人杨杰在中国银行珠海分行申领了中银信用卡后，使用该卡透支消费及取现。被告人杨杰所持信用卡出现逾期未还款，经中国银行珠海分行工作人员多次催收，被告人杨杰超过三个月仍未还款。截止到2014年8月12日，该卡透支本金人民币两万元，利息及其他费用人民币一万元。2014年9月11日，被告人杨杰经公安机关传唤后主动到案，并于次日还清银行全部欠款。上述事实，被告人杨杰在开庭审理过程中亦无异议，且有被害单位中国银行珠海分行代表陈某的陈述，抓获经过、到案经过，户籍资料，授权委托书、报案材料、信用卡申请表，交易明细，被告人杨杰提交的订购合同、采购单、送货单，中国银行珠海分行银行卡部出具的信用卡透支催收说明及欠款明细。
抽取事实：被告人杨杰在中国银行珠海分行申领了中银信用卡后，使用该卡透支消费及取现。被告人杨杰所持信用卡出现逾期未还款，经中国银行珠海分行工作人员多次催收，被告人杨杰超过三个月仍未还款。截止到2014年8月12日，该卡透支本金人民币两万元，利息及其他费用人民币一万元。2014年9月11日，被告人杨杰经公安机关传唤后主动到案，并于次日还清银行全部欠款。

案件文书：```{{文书内容}}```
抽取事实："""

proc_template = """请抽取出所提供案件文书中“公诉机关指控”或“检察院指控”部分的事实。请学习以下一个例子的抽取方式：
案件文书：杨杰信用卡诈骗一审刑事判决书广东省珠海市香洲区人民法院刑事判决书珠香法刑初字第748号：珠海市香洲区人民检察院以珠香检公诉刑诉起诉书指控被告人杨杰犯信用卡诈骗罪，向本院提起公诉。公诉机关指控，被告人杨杰在中国银行珠海分行申领了中银信用卡后，使用该卡透支消费及取现。被告人杨杰所持信用卡出现逾期未还款，经中国银行珠海分行工作人员多次催收，被告人杨杰超过三个月仍未还款。截止到2014年8月12日，该卡透支本金人民币两万元，利息及其他费用人民币一万元。2014年9月11日，被告人杨杰经公安机关传唤后主动到案，并于次日还清银行全部欠款。针对上述指控的犯罪事实，提请本院依法判处。
抽取事实：被告人杨杰在中国银行珠海分行申领了中银信用卡后，使用该卡透支消费及取现。被告人杨杰所持信用卡出现逾期未还款，经中国银行珠海分行工作人员多次催收，被告人杨杰超过三个月仍未还款。截止到2014年8月12日，该卡透支本金人民币两万元，利息及其他费用人民币一万元。2014年9月11日，被告人杨杰经公安机关传唤后主动到案，并于次日还清银行全部欠款。

案件文书：```{{文书内容}}```
抽取事实："""

output_path = './query_fact.json'
if os.path.exists(output_path):
    output_data = json.load(open(output_path, 'r'))
else:
    output_data = {}

court_qids = []
proc_qids = []

def check_response(text):
    if ('杨杰' in text) or (text == ''):
        return False
    return True

for qid in court_data.keys():
    try:
        if not check_response(output_data[qid]):
            court_qids.append(qid)
    except:
        court_qids.append(qid)

for qid in proc_data.keys():
    try:
        if not check_response(output_data[qid]):
            proc_qids.append(qid)
    except:
        proc_qids.append(qid)

for qid in tqdm(court_qids):
    try:
        response = output_data[qid]
    except:
        response = ''
    loop_num = 0
    while (not check_response(response)) and (loop_num < 10):
        loop_num += 1
        input_query = tokenizer.decode(tokenizer(court_data[qid], truncation=True, max_length=2000)['input_ids'], skip_special_tokens=True)
        response, history = model.chat(tokenizer, court_template.replace("{{文书内容}}", input_query), history=None)
    if loop_num == 10:
        print(qid, input_query.split('：', 1)[0])
        
    output_data[qid] = response

    # if len(output_data) % 10 == 0:
    with open(output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

for qid in tqdm(proc_qids):
    try:
        response = output_data[qid]
    except:
        response = ''
    loop_num = 0
    while (not check_response(response)) and (loop_num < 10):
        loop_num += 1
        input_query = tokenizer.decode(tokenizer(proc_data[qid], truncation=True, max_length=2000)['input_ids'], skip_special_tokens=True)
        response, history = model.chat(tokenizer, proc_template.replace("{{文书内容}}", input_query), history=None)
    if loop_num == 10:
        print(qid, input_query.split('：', 1)[0])
        
    output_data[qid] = response

    # if len(output_data) % 10 == 0:
    with open(output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)