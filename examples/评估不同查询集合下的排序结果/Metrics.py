import math

def MAP(ranking_results, rels):
    if type(list(ranking_results.values())[0]) == dict:
        ranking_results = {k: v.keys() for k, v in ranking_results.items()}
    Mean_Average_Precision = 0
    for query_id in ranking_results.keys():
        if all([rels[query_id][did] != 3 for did in rels[query_id].keys()]):
            golden_labels = [2,3]
        else:
            golden_labels = [3]
        num_rel = 0
        Average_Precision = 0
        for i, candidate_id in enumerate(ranking_results[query_id]):
            if candidate_id in rels[query_id].keys() and rels[query_id][candidate_id] in golden_labels:
                num_rel += 1
                Average_Precision += num_rel / (i + 1.0)
        
        if num_rel > 0:
            Average_Precision /= num_rel
        Mean_Average_Precision += Average_Precision
    Mean_Average_Precision /= len(ranking_results.keys())
    return Mean_Average_Precision

def MRR(ranking_results, rels):
    # Calculate Mean Reciprocal Rank (MRR)
    if type(list(ranking_results.values())[0]) == dict:
        ranking_results = {k: v.keys() for k, v in ranking_results.items()}
    Mean_Reciprocal_Rank = 0
    for query_id in ranking_results.keys():
        if all([rels[query_id][did] != 3 for did in rels[query_id].keys()]):
            golden_labels = [2,3]
        else:
            golden_labels = [3]
        Reciprocal_Rank = 0
        for i, candidate_id in enumerate(ranking_results[query_id]):
            if candidate_id in rels[query_id].keys() and rels[query_id][candidate_id] in golden_labels:
                Reciprocal_Rank = 1.0 / (i+1)
                break
        
        Mean_Reciprocal_Rank += Reciprocal_Rank
    
    Mean_Reciprocal_Rank /= len(ranking_results.keys())
    return Mean_Reciprocal_Rank

def Precision_k(ranking_results, rels, k=1):
    # Calculate Precison@k
    if type(list(ranking_results.values())[0]) == dict:
        ranking_results = {k: v.keys() for k, v in ranking_results.items()}
    Precision_k = 0
    for query_id in ranking_results.keys():
        Precision = 0
        if all([rels[query_id][did] != 3 for did in rels[query_id].keys()]):
            golden_labels = [2,3]
        else:
            golden_labels = [3]
        for i, candidate_id in enumerate(ranking_results[query_id]):
            if i+1 <= k:
                if candidate_id in rels[query_id].keys() and rels[query_id][candidate_id] in golden_labels:
                    Precision += 1
            else:
                break
        
        Precision /= k
        Precision_k += Precision
    
    Precision_k /= len(ranking_results.keys())
    return Precision_k

def NDCG_k(ranking_results, rels, k=1):
    # Calculate NDCG@k
    if type(list(ranking_results.values())[0]) == dict:
        ranking_results = {k: v.keys() for k, v in ranking_results.items()}
    Mean_NDCG = 0
    for query_id in ranking_results.keys():
        temp_NDCG = 0.
        temp_IDCG = 0.
        answer_list = sorted([2**(rels[query_id][candidate_id]-1) if (candidate_id in rels[query_id] and rels[query_id][candidate_id] >= 1) else 0. for candidate_id in ranking_results[query_id]], reverse=True)

        for i, candidate_id in enumerate(ranking_results[query_id]):
            if i < k:
                # NDCG
                temp_gain = 2**(rels[query_id][candidate_id]-1) if rels[query_id][candidate_id] >= 1 else 0.
                temp_NDCG += (temp_gain / math.log(i+2, 2))
                
                # IDCG
                temp_IDCG += (answer_list[i] / math.log(i+2, 2))
            else:
                break
        if temp_IDCG > 0:
            Mean_NDCG += (temp_NDCG / temp_IDCG)
    
    Mean_NDCG /= len(ranking_results.keys())
    return Mean_NDCG


# Trainer的compute_metrics函数
def compute_metrics(EvalPrediction):
    init_preds, init_labels = EvalPrediction.predictions, EvalPrediction.label_ids
    result = {}
    preds = {}
    labels = {}
    for i in range(init_labels.shape[0]):
        qid, candidate_id, rel = init_labels[i]
        qid, candidate_id = str(qid), str(candidate_id)
        if qid not in preds.keys():
            preds[qid] = {}
        preds[qid][candidate_id] = float(init_preds[i])
        if qid not in labels.keys():
            labels[qid] = {}
        labels[qid][candidate_id] = rel
    for qid in preds.keys():
        preds[qid] = dict(sorted(preds[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
        
    metrics = 'MAP,MRR,P@1,P@3,NDCG@3,NDCG@5,NDCG@10,NDCG@20,NDCG@30'.split(',')
    for metric in metrics:
        if metric == 'MAP':
            result[f'{metric}'] = MAP(preds, labels)
        elif metric == 'MRR':
            result[f'{metric}'] = MRR(preds, labels)
        elif '@' in metric:
            name, k = metric.split('@')
            if name == 'P':
                result[metric] = Precision_k(preds, labels, k=int(k))
            elif name == 'NDCG':
                result[metric] = NDCG_k(preds, labels, k=int(k))
            else:
                raise NotImplementedError
    return {metric: result[metric] for metric in result.keys()}

def compute_metrics_normal(preds, labels):
    result = {}
    metrics = 'MAP,MRR,P@1,P@3,NDCG@3,NDCG@5,NDCG@10,NDCG@20,NDCG@30'.split(',')
    for metric in metrics:
        if metric == 'MAP':
            result[f'{metric}'] = MAP(preds, labels)
        elif metric == 'MRR':
            result[f'{metric}'] = MRR(preds, labels)
        elif '@' in metric:
            name, k = metric.split('@')
            if name == 'P':
                result[metric] = Precision_k(preds, labels, k=int(k))
            elif name == 'NDCG':
                result[metric] = NDCG_k(preds, labels, k=int(k))
            else:
                raise NotImplementedError
    return {metric: result[metric] for metric in result.keys()}