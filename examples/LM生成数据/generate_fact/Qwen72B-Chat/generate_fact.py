from vllm_wrapper import vLLMWrapper
from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm

# 读入数据
doc_dir = '/share/kelong/chenlong/Legal/data/LeCaRDv2-main/candidate'
docs = {}
for filename in tqdm(os.listdir(doc_dir), ncols=120):
    doc = json.load(open(os.path.join(doc_dir, filename), 'r'))
    docs[filename.split('.')[0]] = doc

key_doc_ids = json.load(open('/share/kelong/chenlong/Legal/Ranking_LeCaRDv2/examples/生成bm25结果/key_ids.json', 'r'))

# 输出文件位置
output_path = './doc_fact.json'
if os.path.exists(output_path):
    output_data = json.load(open(output_path, 'r'))
else:
    output_data = {}
    
# check_response
def check_response(text):
    # 要么以“与该罪名相关的犯罪情节是：”开头，要么以“该案件中没有与该罪名相关的犯罪情节，原因是”开头
    if text.split('：', 1)[0].endswith('起因、经过、结果分别为') or text.startswith('该案件中没有'):
        return True
    else:
        return False
    
# 确定哪些数据需要进行生成
did_crimes = []
for did in key_doc_ids:
    for crime in docs[did]['charge']:
        try:
            if check_response(output_data[did][crime]) != True:
                did_crimes.append([did, crime])
        except:
            did_crimes.append([did, crime])

# prompt template
prompt_template = """```{{案件文本}}```\n请以“与该罪名相关的犯罪情节的起因、经过、结果分别为：”作为输出的开头，用自己的语言分别总结以上案件与“{{罪名}}”相关的起因、经过、结果，每一部分都保证在50字内。如果文本中不包含该罪名的犯罪情节，请直接输出“该案件中没有与该罪名相关的犯罪情节，原因是：[你的原因]”"""

# 读入model
model_dir = "Qwen/Qwen-72B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vllm_model = vLLMWrapper(model_dir, quantization='gptq', dtype="float16", gpu_memory_utilization=0.9, tensor_parallel_size=4)

# 开始推理
for i, (did, crime) in enumerate(tqdm(did_crimes, ncols=120)):
    try:
        response = output_data[did][crime]
    except:
        response = ''
    loop_num = 0
    while (not check_response(response)) and (loop_num < 1):
        loop_num += 1
        case_text = tokenizer.decode(tokenizer(docs[did]['fact'], truncation=True, max_length=2000)['input_ids'], skip_special_tokens=True)
        response, history = vllm_model.chat(prompt_template.replace("{{案件文本}}", case_text).replace("{{罪名}}", crime), history=None)
    # if loop_num == 1:
    if not check_response(response):
        print(did, crime)
        
    if did not in output_data.keys():
        output_data[did] = {}
    output_data[did][crime] = response

    if (i % 10 == 0) or ((i+1) == len(did_crimes)):
        with open(output_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
