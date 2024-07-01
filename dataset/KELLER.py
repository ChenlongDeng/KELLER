import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from itertools import chain
import numpy as np

# CustomDataset Class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, querys, candidates, labels, tokenizer, data_mode):
        super().__init__()
        self.max_crime_num = args.max_crime_num
        self.max_fact_len = args.max_fact_len
        self.querys = querys
        self.candidates = candidates
        self.labels = labels
        self.tokenizer = tokenizer
        self.data_mode = data_mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        query_str_list = [''] * self.max_crime_num
        query_seq_mask = [0] * self.max_crime_num
        query = [(k, v) for k, v in self.querys[self.labels[idx]['qid']].items()]
        for i in range(min(len(query_str_list), len(query))):
            query_str_list[i] = f'{query[i][0]}的相关情节：{query[i][1]}'
            query_seq_mask[i] = 1
        
        # query_str_all_crime
        query_str_max_len = 512 // len([s for s in query_str_list if s != ''])
        query_str_all_crime = [''.join([s[:query_str_max_len] for s in query_str_list])]
        
        doc_str_list = []
        doc_seq_mask = []
        for d_pos, did in enumerate(self.labels[idx]['did']):
            temp_doc_str_list = [''] * self.max_crime_num
            temp_doc_seq_mask = [0] * self.max_crime_num
            temp_doc_crimes = [''] * self.max_crime_num
            doc = [(k, v) for k, v in self.candidates[did].items()]
            for i in range(min(len(temp_doc_str_list), len(doc))):
                temp_doc_str_list[i] = f'{doc[i][0]}的相关情节：{doc[i][1]}'
                temp_doc_crimes[i] = doc[i][0]
                temp_doc_seq_mask[i] = 1
            doc_str_list.extend(temp_doc_str_list)
            doc_seq_mask.extend(temp_doc_seq_mask)
            
            if d_pos == 0:
                doc_crimes = list(self.candidates[did].keys())[:len(temp_doc_str_list)]
                fine_grained_label = [-1] * self.max_crime_num
                for i, q in enumerate(query):
                    if i >= len(query_str_list):
                        break
                    if (q[0] != '') and (q[0] in doc_crimes):
                        fine_grained_label[i] = doc_crimes.index(q[0])
        
        if self.data_mode == 'Train':
            return {
                'qid': [self.labels[idx]['qid']],
                'query_str_list': query_str_list,
                'query_seq_mask': query_seq_mask,
                'query_str_all_crime': query_str_all_crime,
                'fine_grained_label': fine_grained_label,
                'doc_str_list': doc_str_list,
                'doc_seq_mask': doc_seq_mask
            }
        elif self.data_mode == 'Test':
            labels = [int(self.labels[idx]['qid']), int(self.labels[idx]['did'][0]), self.labels[idx]['rel']]
            return {
                'query_str_list': query_str_list,
                'query_seq_mask': query_seq_mask,
                'query_str_all_crime': query_str_all_crime,
                'doc_str_list': doc_str_list,
                'doc_seq_mask': doc_seq_mask,
                'labels': torch.tensor(labels, dtype=torch.long)
            }

# CustomDataCollator Class
class CustomDataCollator:
    def __init__(self, args, tokenizer):
        self.tokenizer = tokenizer
        self.max_crime_num = args.max_crime_num
        self.max_fact_len = args.max_fact_len
    
    def __call__(self, batch):
        if 'labels' in batch[0].keys():
            labels = torch.stack([d['labels'] for d in batch], dim=0)
        else:
            labels = None
            
        batch = {k: list(chain(*[d[k] for d in batch])) for k in batch[0] if k != 'labels'}
        query_input = self.tokenizer(batch['query_str_list'], padding='max_length', max_length=self.max_fact_len, truncation=True, return_tensors='pt')
        doc_input = self.tokenizer(batch['doc_str_list'], padding='max_length', max_length=self.max_fact_len, truncation=True, return_tensors='pt')
        query_str_all_crime_input = self.tokenizer(batch['query_str_all_crime'], padding='max_length', max_length=self.max_fact_len, truncation=True, return_tensors='pt')
        
        if labels == None:
            batch_size = int(len(batch['query_str_list']) / self.max_crime_num)
            doc_num_per_query = int(len(batch['doc_str_list']) / len(batch['query_str_list']))
            contrastive_mask = torch.ones(len(batch['qid']), len(batch['qid'])*doc_num_per_query)
            for i in range(len(batch['qid'])):
                for j in range(len(batch['qid'])):
                    if i == j:
                        continue
                    if batch['qid'][i] == batch['qid'][j]:
                        contrastive_mask[i][j*doc_num_per_query] = 0
            
            fine_grained_mask = torch.ones(len(batch['query_str_list']), len(batch['doc_str_list']))
            query_crimes = [i.split('的相关情节：', 1)[0] for i in batch['query_str_list']]
            doc_crimes = [i.split('的相关情节：', 1)[0] for i in batch['doc_str_list']]
            for i, query_crime in enumerate(query_crimes):
                start_idx = (i//self.max_crime_num)*doc_num_per_query*self.max_crime_num
                if query_crime == '':
                    continue
                for j, doc_crime in enumerate(doc_crimes):
                    if (j >= start_idx) and (j < (start_idx+self.max_crime_num)):
                        continue
                    if query_crime == doc_crime:
                        fine_grained_mask[i][j] = 0
            
            return {
                **{'query_'+k: v.squeeze() for k, v in query_input.items()},
                **{'doc_'+k: v.squeeze() for k, v in doc_input.items()},
                **{'query_all_crime_'+k: v.squeeze() for k, v in query_str_all_crime_input.items()},
                'fine_grained_label': torch.tensor(batch['fine_grained_label']),
                'contrastive_mask': contrastive_mask,
                'fine_grained_mask': fine_grained_mask,
                'query_seq_mask': torch.tensor(batch['query_seq_mask'], dtype=torch.float),
                'doc_seq_mask': torch.tensor(batch['doc_seq_mask'], dtype=torch.float)
            }
        else:
            return {
                **{'query_'+k: v.squeeze() for k, v in query_input.items()},
                **{'doc_'+k: v.squeeze() for k, v in doc_input.items()},
                **{'query_all_crime_'+k: v.squeeze() for k, v in query_str_all_crime_input.items()},
                'query_seq_mask': torch.tensor(batch['query_seq_mask'], dtype=torch.float),
                'doc_seq_mask': torch.tensor(batch['doc_seq_mask'], dtype=torch.float),
                'labels': labels
            }

# Data_Provider Class
class Data_Provider():
    def __init__(self, args):
        # Basic attr
        self.tokenizer = AutoTokenizer.from_pretrained(args.PLM_path)
        self.Model_name = args.Model_name

        # Data list
        self.labels_train = []
        self.labels_test_LeCaRD = []
        self.labels_test_LeCaRDv2 = []
        
        print('Preprocessing data...')
        self.Prepare_data(args)
        # Initialize torch dataset and dataloader
        print('Building dataset...')
        self.dataset_train = CustomDataset(args=args, querys=self.LeCaRDv2_train_querys, candidates=self.LeCaRDv2_decomposed_candidates, labels=self.labels_train, tokenizer=self.tokenizer, data_mode='Train')
        self.dataset_test_LeCaRDv2 = CustomDataset(args=args, querys=self.LeCaRDv2_test_querys, candidates=self.LeCaRDv2_decomposed_candidates, labels=self.labels_test_LeCaRDv2, tokenizer=self.tokenizer, data_mode='Test')
        self.dataset_test_LeCaRD = CustomDataset(args=args, querys=self.LeCaRD_querys, candidates=self.LeCaRD_candidates, labels=self.labels_test_LeCaRD, tokenizer=self.tokenizer, data_mode='Test')
        self.data_collator = CustomDataCollator(args, self.tokenizer)
    
    def check_response(self, text):
        if text.split('：', 1)[0].endswith('起因、经过、结果分别为') and not all([i in text for i in ['起因：无', '经过：无', '结果：无']]) and not ('抱歉' in text):
            return True
        else:
            return False
    
    # Main entry of prepare dataset
    def Prepare_data(self, args):
        # LeCaRD: 只用来测试
        self.LeCaRD_querys = json.load(open(args.LeCaRD_query_path, 'r'))
        self.LeCaRD_candidates = json.load(open(args.LeCaRD_candidate_path, 'r'))
        for qid in self.LeCaRD_querys.keys():
            self.LeCaRD_querys[qid] = self.LeCaRD_querys[qid]['q']
        for did in self.LeCaRD_candidates.keys():
            self.LeCaRD_candidates[did] = self.LeCaRD_candidates[did]['ajjbqk']
        
        if "case_decomposition" in vars(args) and args.case_decomposition:
            self.LeCaRD_querys = json.load(open(args.LeCaRD_decomposed_query_path, 'r'))
            self.LeCaRD_candidates = json.load(open(args.LeCaRD_decomposed_doc_path, 'r'))
            for qid in self.LeCaRD_querys.keys():
                self.LeCaRD_querys[qid] = {k: v.split('：', 1)[-1] for k, v in self.LeCaRD_querys[qid].items() if self.check_response(v)}
            for did in self.LeCaRD_candidates.keys():
                self.LeCaRD_candidates[did] = {k: v.split('：', 1)[-1] for k, v in self.LeCaRD_candidates[did].items() if self.check_response(v)}
            
        for qid in ['6816', '2403']:
            self.LeCaRD_querys.pop(qid)
        
        # LeCaRDv2: 划分为训练集和测试集
        self.LeCaRDv2_train_querys = {}
        self.LeCaRDv2_test_querys = {}
        with open(args.LeCaRDv2_train_query_path, 'r') as f:
            for line in f:
                query = json.loads(line)
                try:
                    self.LeCaRDv2_train_querys[str(query['id'])] = query['fact']
                except:
                    pass
        with open(args.LeCaRDv2_test_query_path, 'r') as f:
            for line in f:
                query = json.loads(line)
                if str(query['id']) != '20':
                    continue
                self.LeCaRDv2_test_querys[str(query['id'])] = query['fact']
        if "case_decomposition" in vars(args) and args.case_decomposition:
            self.LeCaRDv2_decomposed_querys = json.load(open(args.LeCaRDv2_decomposed_query_path, 'r'))
            self.LeCaRDv2_decomposed_candidates = json.load(open(args.LeCaRDv2_decomposed_doc_path, 'r'))
            for qid in list(self.LeCaRDv2_train_querys.keys()):
                try:
                    self.LeCaRDv2_train_querys[qid] = {k: v.split('：', 1)[-1] for k, v in self.LeCaRDv2_decomposed_querys[qid].items() if self.check_response(v)}
                except:
                    self.LeCaRDv2_train_querys.pop(qid)
            for qid in self.LeCaRDv2_test_querys.keys():
                if qid != '20':
                    continue
                self.LeCaRDv2_test_querys[qid] = {k: v.split('：', 1)[-1] for k, v in self.LeCaRDv2_decomposed_querys[qid].items() if self.check_response(v)}
            for did in self.LeCaRDv2_decomposed_candidates.keys():
                self.LeCaRDv2_decomposed_candidates[did] = {k: v.split('：', 1)[-1] for k, v in self.LeCaRDv2_decomposed_candidates[did].items() if self.check_response(v)}
        else:
            self.LeCaRDv2_candidates = {}
            for filename in tqdm(os.listdir(args.LeCaRDv2_candidate_path), ncols=120, desc='reading LeCaRDv2 docs'):
                did = filename.split('.')[0]
                self.LeCaRDv2_candidates[did] = json.load(open(os.path.join(args.LeCaRDv2_candidate_path, filename), 'r'))['fact']
        
        self.LeCaRDv2_bm25 = json.load(open(args.LeCaRDv2_bm25_path, 'r'))
        
        # 两个数据集的label
        self.LeCaRD_label = json.load(open(args.LeCaRD_label_path, 'r'))
        self.LeCaRDv2_label = {}
        with open(args.LeCaRDv2_label_path, 'r') as f:
            for line in f:
                qid, _, did, rel = line.strip().split('\t')
                if qid not in self.LeCaRDv2_label.keys():
                    self.LeCaRDv2_label[qid] = {}
                self.LeCaRDv2_label[qid][did] = int(rel)
                
        # 训练集是LeCaRDv2，每个query配对一个正例，可以增加强负例
        for qid in self.LeCaRDv2_train_querys.keys():
            if all([self.LeCaRDv2_label[qid][did] != 3 for did in self.LeCaRDv2_label[qid].keys()]):
                golden_labels = [2,3]
            else:
                golden_labels = [3]
                
            for did in self.LeCaRDv2_label[qid].keys():
                if self.LeCaRDv2_label[qid][did] in golden_labels:
                    if args.hard_negative_num == 0:
                        self.labels_train.append(
                            {
                                'qid': qid,
                                'did': [did]
                            }
                        )
                    else:
                        negatives = [i for i in self.LeCaRDv2_bm25[qid].keys() if (i not in self.LeCaRDv2_label[qid].keys()) or (self.LeCaRDv2_label[qid][i] not in golden_labels)]
                        candidates = [did] + negatives[10:10+args.hard_negative_num]
                        self.labels_train.append(
                            {
                                'qid': qid,
                                'did': candidates
                            }
                        )

        # 测试集用两个，分别是LeCaRD和LeCaRDv2
        # 对LeCaRDv2中没有区分性标注的query，从bm25top100-110取10个加进来
        for qid in self.LeCaRD_querys.keys():
            for did in self.LeCaRD_label[qid].keys():
                rel = self.LeCaRD_label[qid][did] if did in self.LeCaRD_label[qid].keys() else 0
                self.labels_test_LeCaRD.append(
                    {
                        'qid': qid,
                        'did': [did],
                        'rel': rel
                    }
                )
            
                
        for qid in self.LeCaRDv2_test_querys.keys():
            for did in self.LeCaRDv2_label[qid].keys():
                # For debug
                # if did != '3189118':
                #     continue
                rel = self.LeCaRDv2_label[qid][did] if did in self.LeCaRDv2_label[qid].keys() else 0
                self.labels_test_LeCaRDv2.append(
                    {
                        'qid': qid,
                        'did': [did],
                        'rel': rel
                    }
                )
            # if all([self.LeCaRDv2_label[qid][did] != 3 for did in self.LeCaRDv2_label[qid].keys()]):
            #     golden_labels = [2,3]
            # else:
            #     golden_labels = [3]
            # if all(self.LeCaRDv2_label[qid][did] in golden_labels for did in self.LeCaRDv2_label[qid].keys()):
            #     extra_candidates = [i for i in self.LeCaRDv2_bm25[qid].keys() if (i not in self.LeCaRDv2_label[qid].keys()) or (self.LeCaRDv2_label[qid][i] not in golden_labels)]
            #     for did in extra_candidates[100:110]:
            #         self.labels_test_LeCaRDv2.append(
            #             {
            #                 'qid': qid,
            #                 'did': [did],
            #                 'rel': 0
            #             }
            #         )
            
                
    def get_dataset(self):
        return self.dataset_train, self.dataset_test_LeCaRD, self.dataset_test_LeCaRDv2

    
    def get_data_collator(self):
        return self.data_collator
