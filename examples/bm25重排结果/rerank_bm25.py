import thulac   
import json
from tqdm import tqdm
import os
from rank_bm25 import BM25Okapi
from Metrics import compute_metrics_normal

# 初始化THULAC实例
thu1 = thulac.thulac(seg_only=True)  # 设置seg_only=True只进行分词，不进行词性标注
querys = {}
with open('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/train_query.jsonl', 'r') as f:
    for line in f:
        query = json.loads(line)
        querys[str(query['id'])] = query['fact']
        
test_querys = []
with open('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/query/test_query.jsonl', 'r') as f:
    for line in f:
        query = json.loads(line)
        querys[str(query['id'])] = query['fact']
        test_querys.append(str(query['id']))

labels = {}
with open('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/label/relevence.trec', 'r') as f:
    for line in f:
        qid, _, did, rel = line.split('\t')
        if qid not in labels.keys():
            labels[qid] = {}
        labels[qid][did] = int(rel)
        
LeCaRDv2_candidates = {}
for filename in tqdm(os.listdir('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/candidate/'), ncols=120, desc='reading LeCaRDv2 docs'):
    did = filename.split('.')[0]
    LeCaRDv2_candidates[did] = json.load(open(os.path.join('/share/kelong/chenlong/Legal/data/LeCaRDv2-main/candidate/', filename), 'r'))['fact']
dids = list(LeCaRDv2_candidates.keys())

LeCaRDv2_bm25 = json.load(open('/share/kelong/chenlong/Legal/Ranking_LeCaRDv2/examples/生成bm25结果/bm25_top1k.json', 'r'))

# 需要一个q_candidates
q_candidates = {}
for qid in test_querys:
    q_candidates[qid] = list(labels[qid].keys())
    if all([labels[qid][did] != 3 for did in labels[qid].keys()]):
        golden_labels = [2,3]
    else:
        golden_labels = [3]
    if all(labels[qid][did] in golden_labels for did in labels[qid].keys()):
        extra_candidates = [i for i in LeCaRDv2_bm25[qid].keys() if (i not in labels[qid].keys()) or (labels[qid][i] not in golden_labels)]
        for did in extra_candidates[100:110]:
            q_candidates[qid].append(did)
            
# 中文文档集合
# 使用THULAC进行分词
corpus = []
for doc in tqdm(LeCaRDv2_candidates.values()):
    corpus.append(thu1.cut(doc, text=True).split(' '))

# 创建BM25模型
bm25 = BM25Okapi(corpus)

result = {}
# 查询（同样需要分词）
for qid in tqdm(test_querys):
    query = querys[qid]
    query_words = thu1.cut(query, text=True).split(' ')
    scores = bm25.get_scores(query_words)
    result[qid] = {did: scores[i] for i, did in enumerate(dids)}
    result[qid] = {did: result[qid][did] for did in result[qid].keys() if did in q_candidates[qid]}
    result[qid] = dict(sorted(result[qid].items(), key=lambda item: item[1], reverse=True))
    
with open('./rank_bm25.json', 'w') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(compute_metrics_normal(result, labels))