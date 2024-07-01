import json
import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# 读入candidates和key_reasoning
# 将key_reasoning拼接在candidate前面
key_reasoning = json.load(open('./key_reasoning.json', 'r'))
docs = {}
doc_path = '/share/kelong/chenlong/Legal/data/LeCaRDv2-main/candidate'
for filename in tqdm(os.listdir(doc_path)):
    doc = json.load(open(os.path.join(doc_path, filename), 'r'))
    did = filename.split('.')[0]
    docs[did] = ''.join(key_reasoning[did].values())
    

# 生成一维的embedding
model = AutoModel.from_pretrained('/share/kelong/chenlong/LMs/SAILER').cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('/share/kelong/chenlong/LMs/SAILER')

class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, docs):
        self.docs = docs
        self.did_list = list(self.docs.keys())
    
    def __len__(self):
        return len(self.did_list)

    def __getitem__(self, idx):
        input_tensor = tokenizer(self.docs[self.did_list[idx]], padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        return self.did_list[idx], input_tensor['input_ids'].squeeze(), input_tensor['attention_mask'].squeeze()

dataset = torch_dataset(docs)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, num_workers=8)

result = {}
with torch.no_grad():
    for dids, input_ids, attention_mask in tqdm(dataloader):
        output_tensors = model(**{'input_ids': input_ids.to('cuda'), 'attention_mask': attention_mask.to('cuda')})['last_hidden_state'][:, 0, :]
        for i, did in enumerate(dids):
            result[did] = output_tensors[i].cpu()

# 打包为pt文件
with open('./only_key_reasoning_embedding.pkl', 'wb') as f:
    pickle.dump(result, f)