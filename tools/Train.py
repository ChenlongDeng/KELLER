import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup, TrainingArguments, Trainer
from tqdm import tqdm
from .Test import test
from .Metrics import compute_metrics, save_training_results, save_eva_files

def train(args, model, data_provider, load=False):
    if args.Model_name in ['BERT', 'RoBERTa', 'BGE', 'SAILER', 'Lawformer', 'BERTPLI', 'KELLER']:
        DualEncoder_train(args, model, data_provider, load=load)
    else:
        raise NotImplementedError

def DualEncoder_train(args, model, data_provider, load=False):
    training_args = TrainingArguments(**args.training_args)
    train_dataset, LeCaRD_test_dataset, LeCaRDv2_test_dataset = data_provider.get_dataset()
    data_collator = data_provider.get_data_collator()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=LeCaRDv2_test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    
    # Zeroshot能力测试
    if args.test_zeroshot:
        LeCaRDv2_zeroshot_output = trainer.predict(test_dataset=LeCaRDv2_test_dataset)
        LeCaRD_zeroshot_output = trainer.predict(test_dataset=LeCaRD_test_dataset)
        save_eva_files(args, LeCaRDv2_zeroshot_output, 'LeCaRDv2', zeroshot=True)
        save_eva_files(args, LeCaRD_zeroshot_output, 'LeCaRD', zeroshot=True)
    # trainer.evaluate()
    trainer.train()
    
    LeCaRDv2_output = trainer.predict(test_dataset=LeCaRDv2_test_dataset)
    LeCaRD_output = trainer.predict(test_dataset=LeCaRD_test_dataset)
    if int(args.local_rank) == 0:
        # 生成最佳结果对应的logits
        save_eva_files(args, LeCaRDv2_output, 'LeCaRDv2')
        save_eva_files(args, LeCaRD_output, 'LeCaRD')
        # 将训练结果导出到results.txt中
        if args.test_zeroshot:
            save_training_results(args, compute_metrics(LeCaRDv2_output), compute_metrics(LeCaRD_output), compute_metrics(LeCaRDv2_zeroshot_output), compute_metrics(LeCaRD_zeroshot_output))
        else:
            save_training_results(args, compute_metrics(LeCaRDv2_output), compute_metrics(LeCaRD_output))
        