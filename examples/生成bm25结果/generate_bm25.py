import thulac   
import json
from tqdm import tqdm
import os
from rank_bm25 import BM25Okapi

# 初始化THULAC实例
thu1 = thulac.thulac(seg_only=True)  # 设置seg_only=True只进行分词，不进行词性标注
querys = {}
with open('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/train_query.jsonl', 'r') as f:
    for line in f:
        query = json.loads(line)
        querys[str(query['id'])] = query['fact']
        
with open('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/test_query.jsonl', 'r') as f:
    for line in f:
        query = json.loads(line)
        querys[str(query['id'])] = query['fact']
        
LeCaRDv2_candidates = {}
for filename in tqdm(os.listdir('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/candidate/'), ncols=120, desc='reading LeCaRDv2 docs'):
    did = filename.split('.')[0]
    LeCaRDv2_candidates[did] = json.load(open(os.path.join('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/candidate/', filename), 'r'))['fact']
dids = list(LeCaRDv2_candidates.keys())
# 中文文档集合
# 使用THULAC进行分词
corpus = [thu1.cut(doc, text=True).split(' ') for doc in LeCaRDv2_candidates.values()]

# 创建BM25模型
bm25 = BM25Okapi(corpus)

result = {}
# 查询（同样需要分词）
for qid, query in tqdm(querys.items()):
    query_words = thu1.cut(query, text=True).split(' ')
    scores = bm25.get_scores(query_words)
    result[qid] = {did: scores[i] for i, did in enumerate(dids)}
    result[qid] = dict(sorted(result[qid].items(), key=lambda item: item[1], reverse=True)[:1000])
    
with open('./bm25_top1k.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)