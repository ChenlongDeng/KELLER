from vllm_wrapper import vLLMWrapper
from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm

# 读入数据
query_path = '/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/query.json'
querys = {}
with open(query_path, 'r') as f:
    for line in f:
        query = json.loads(line)
        querys[str(query['id'])] = query['query']

# 输出文件位置
output_path = './extracted_query_crimes.json'
if os.path.exists(output_path):
    output_data = json.load(open(output_path, 'r'))
else:
    output_data = {}
    
# check_response
def check_response(text):
    # 要么以“与该罪名相关的犯罪情节是：”开头，要么以“该案件中没有与该罪名相关的犯罪情节，原因是”开头
    if text.split('：', 1)[0].endswith('该法律文书中检察院指控的罪名有') and all([i.endswith('罪') for i in text.split('：', 1)[1].split('；')]):
        return True
    else:
        return False
    
# 确定哪些数据需要进行生成
qids = []
for qid in querys.keys():
    try:
        if check_response(output_data[qid]) != True:
            qids.append(qid)
    except:
        qids.append(qid)

# prompt template
prompt_template = """```{{案件文本}}```"""
system_prompt_template = """你现在是一位中国法律专家，擅长从冗长的法律文书中抽取检察院指控的罪名。接下来，用户将会输入一段冗长的法律案件文本，该案件可能包含多个罪名的情节，你的任务是从文书中找到检察院指控的所有罪名。现在请你以“该法律文书中检察院指控的罪名有：”作为输出的开头，输出检察院指控的所有罪名，不同罪名之间用中文分号（；）分隔开，请不要重复输出相同的罪名"""

# 读入model
model_dir = "Qwen/Qwen-72B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vllm_model = vLLMWrapper(model_dir, quantization='gptq', dtype="float16", gpu_memory_utilization=0.9, tensor_parallel_size=4)

# 开始推理
for i, qid in enumerate(tqdm(qids, ncols=120)):
    try:
        response = output_data[qid]
    except:
        response = ''
    loop_num = 0
    while (not check_response(response)) and (loop_num < 3):
        loop_num += 1
        case_text = tokenizer.decode(tokenizer(querys[qid], truncation=True, max_length=8192)['input_ids'], skip_special_tokens=True)
        response, history = vllm_model.chat(prompt_template.replace("{{案件文本}}", case_text), history=None, system=system_prompt_template)
    # if loop_num == 1:
    if not check_response(response):
        print(qid)
        
    if qid not in output_data.keys():
        output_data[qid] = {}
    output_data[qid] = response

    if (i % 10 == 0) or ((i+1) == len(qids)):
        with open(output_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
