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
query_crimes = '/share/kelong/chenlong/Legal/Ranking_LeCaRDv2/examples/LM生成数据/generate_key_reasoning/Qwen14B-chat/LeCaRDv2_query_crimes_labeled.json'
querys = {}
with open(query_path, 'r') as f:
    for line in f:
        query = json.loads(line)
        querys[str(query['id'])] = query['fact']
        querys['charge'] = query_crimes[str(query['id'])]

output_path = f'./LeCaRDv2_query_key_reasoning.json'
if os.path.exists(output_path):
    output_data = json.load(open(output_path, 'r'))
else:
    output_data = {}


template = """请基于给定的法律文书标题，生成该标题中包含的罪名（可能有一个或多个，多个时用分号隔开）。
标题：2010年7月至2011年9月，被告人钱治友、包震军、何军伙同赖赞亦、汤寿祥、包震坚（三人均另案处理）在湘潭市雨湖区基建营附近开设“魔幻战神”电游室，经营具有赌博功能的万能鲨鱼机、赛车机、金镶玉等赌博电游机进行赌博。该电游赌场的股份情况为：提供赌博机器的广州迅捷公司占整个电游室收益的45％，余下的55％再按100％分配，其中包震军、钱治友各占37.5％，何军占15％，赖赞亦占10％。汤寿祥在电游室负责抄每台游戏机的输赢情况，统计营业额，包震军则要其弟包震坚任电游室法定代表人并负责日常管理，处理与各管理部门的关系，赖赞亦负责电游室的账务。该赌场营业至2011年9月，被告人钱治友、包震军分别从中非法获利30万余元（人民币，下同），被告人何军从中非法获利60万余元。
包含罪名：开设赌场罪
罪名认定部分：被告人以营利为目的，在有合法经营资格的电子游戏厅中设置具有赌博功能的电子游戏机，非法获利数十万元，其行为均构成开设赌场罪。

案情：被告人罗华林提议殴打一辆摩托车上的人，同案人庞峰新驾驶一辆面包车载罗华林、同案犯庞培仪、庞康志、庞世康尾随该摩托车，后被害人钟某华、罗某凡、肖某勇发现被跟踪加速逃跑。庞峰新加速追赶，罗华林等四人挥刀喝令对方停车。追赶到廉江市廉江大道中小龄童服装店附近路段，摩托车躲避不及，连人带车撞到路边路基，面包车撞倒逆向骑一辆三轮车的被害人许某平、骑一辆自行车的被害人吴某某。造成被害人许某平死亡，钟某华、罗某凡轻伤，肖某勇、吴某某轻微伤。 2018年7月4日，被告人罗华林近亲属赔偿人民币4.2万元给许某平的家属，许的家属对罗华林予以谅解。 2017年6月29日0时许，被告人罗华林伙同冯炳涛、庞弟、黄宏麟（均已判刑）及李某允、刘某锐、黄某原、李某熙等人（均另案处理）及许某红到廉江市同济市场门口吃夜宵，许某红与隔壁桌的陈某永互搂着喝酒。罗华林等人对此不满，持铁台脚、胶凳、木棍、剪刀殴打被害人龙某达、龙某鸿致其轻微伤。
罪名：以危险方法危害公共安全罪
罪名认定部分：被告人同他人在城市主干道公共道路上超速驾驶追赶拦截被害人，开车撞向车流人流密集的不特定人，并持刀砍向被追赶的被害人，放任人员伤亡后果的发生，其行为已构成以危险方法危害公共安全罪。

案情：```{{案情}}```
罪名：{{罪名}}
请不要生成其他罪名的描述
罪名认定部分："""

problem_names = ['钱治友', '罗华林']
def check_response(qid, crime, text):
    other_crimes = set(querys[qid]['charge']) - set([crime])
    if any([i in text for i in problem_names]) or all([sub_crime not in text for sub_crime in crime.split('、')]) or any([other_crime in text for other_crime in other_crimes if other_crime not in crime]):
        return False
    else:
        return True
qids_crimes = []
for qid in querys.keys():
    for crime in querys[qid]['charge']:
        crime = crime.replace('(', '（').replace(')', '）')
        try:
            if not check_response(qid, crime, output_data[qid][crime]):
                qids_crimes.append([qid, crime])
        except:
            qids_crimes.append([qid, crime])
# if args.rank == 0:
#     qids = qids[:int(len(qids)/2)]
# else:
#     qids = qids[int(len(qids)/2):]
    
for qid, crime in tqdm(qids_crimes):
    # if qid not in extracted_crimes.keys():
    #     continue
    # for crime in extracted_crimes[qid]:
        # if (qid in output_data.keys()) and (crime in output_data[qid].keys()) and all([i not in output_data[qid][crime] for i in ['抱歉', '没有提到', '没有涉及']]):
        #     continue
    try:
        response = output_data[qid][crime]
    except:
        response = ''
    loop_num = 0
    while (not check_response(qid, crime, response)) and (loop_num < 10):
        loop_num += 1
        fact = tokenizer.decode(tokenizer(querys[qid]['fact'], truncation=True, max_length=1500)['input_ids'], skip_special_tokens=True)
        response, history = model.chat(tokenizer, template.replace("{{案情}}", fact).replace("{{罪名}}", crime), history=None)
    if loop_num == 10:
        print(qid, crime)
    if qid not in output_data.keys():
        output_data[qid] = {}
    output_data[qid][crime] = response

    # if len(output_data) % 10 == 0:
    with open(output_path, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

