from vllm_wrapper import vLLMWrapper
from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm

# 读入数据
query_path = '/share/kelong/chenlong/Legal/Ranking_LeCaRDv2/examples/LeCaRDv2查询质量检验/LeCaRDv2_querys_labeled.json'
querys = json.load(open(query_path, 'r'))
extracted_query_crimes = json.load(open('/share/kelong/chenlong/Legal/Ranking_LeCaRDv2/examples/LM生成数据/generate_fact/Qwen72B-Chat/extracted_query_crimes.json', 'r'))

original_query_path = '/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/query.json'
original_querys = {}
with open(original_query_path, 'r') as f:
    for line in f:
        query = json.loads(line)
        original_querys[str(query['id'])] = query['query']
# 输出文件位置
output_path = './query_fact.json'
if os.path.exists(output_path):
    output_data = json.load(open(output_path, 'r'))
else:
    output_data = {}
    
# check_response
def check_response(text):
    # 以“起因、经过、结果分别为”开头的是正确的
    # “该案件中没有与”开头，“抱歉”，“没有提到”在文中：属于没有找到相应情节，应该在qw中再找一次（qw）
    # 剩下的都是不符合格式的，直接在fact中再找一次（fact）
    if text.split('：', 1)[0].endswith('起因、经过、结果分别为') and not all([i in text for i in ['起因：无', '经过：无', '结果：无']]) and not ('抱歉' in text):
        return True
    elif text.startswith('该案件中没有与') or ('抱歉' in text) or ('没有提到' in text):
        return 'qw'
    else:
        return 'qw'
    
# 确定哪些数据需要进行生成
qids = []
for qid in querys.keys():
    crimes = list(set(extracted_query_crimes[qid].split('该法律文书中检察院指控的罪名有：', 1)[1].split('；')))
    for crime in crimes:
        try:
            if check_response(output_data[qid][crime]) != True:
                qids.append([qid, crime])
        except:
            qids.append([qid, crime])

# prompt template
prompt_template = """```{{案件文本}}```"""
system_prompt_template = """你现在是一位中国法律专家，擅长从冗长的法律文本中总结关键事实。用户将会输入一段冗长法律案件文本，该案件可能包含多个罪名的情节，你的任务是从案件文本中总结与其中某一个罪名相关的犯罪情节。现在请你以“与该罪名相关的犯罪情节的起因、经过、结果分别为：”作为输出的开头，用自己的语言分别总结案件文本中与“{{罪名}}”相关的起因、经过、结果，每一部分都保证在50字内"""

# 读入model
model_dir = "Qwen/Qwen-72B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vllm_model = vLLMWrapper(model_dir, quantization='gptq', dtype="float16", gpu_memory_utilization=0.9, tensor_parallel_size=4)

# 开始推理
for i, (qid, crime) in enumerate(tqdm(qids, ncols=120)):
    try:
        response = output_data[qid][crime]
    except:
        response = ''
    loop_num = 0
    while (check_response(response) != True) and (loop_num < 20):
        loop_num += 1
        case_text = tokenizer.decode(tokenizer(original_querys[qid], truncation=True, max_length=8192)['input_ids'], skip_special_tokens=True)
        response, history = vllm_model.chat(prompt_template.replace("{{案件文本}}", case_text), history=None, system=system_prompt_template.replace("{{罪名}}", crime))
    # if loop_num == 1:
    if check_response(response) != True:
        print(qid, crime)
        
    if qid not in output_data.keys():
        output_data[qid] = {}
    output_data[qid][crime] = response

    if (i % 10 == 0) or ((i+1) == len(qids)):
        with open(output_path, 'w') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
