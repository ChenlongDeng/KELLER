import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
from transformers import Trainer, TrainingArguments
from .Metrics import compute_metrics, save_eva_files, save_training_results

def test(args, model, data_provider, load=False):
    if args.Model_name in ['BERT', 'RoBERTa', 'BGE', 'SAILER', 'Lawformer', 'BERTPLI']:
        DualEncoder_test(args, model, data_provider, load=load)
    elif args.Model_name == 'KELLER':
        KELLER_test(args, model, data_provider, load=load)
    else:
        raise NotImplementedError


def DualEncoder_test(args, model, data_provider, load=False):
    # Function for testing model
    training_args = TrainingArguments(**args.training_args)
    train_dataset, LeCaRD_test_dataset, LeCaRDv2_test_dataset = data_provider.get_dataset()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=LeCaRDv2_test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Zeroshot能力测试
    if args.test_zeroshot:
        LeCaRDv2_zeroshot_output = trainer.predict(test_dataset=LeCaRDv2_test_dataset)
        LeCaRD_zeroshot_output = trainer.predict(test_dataset=LeCaRD_test_dataset)
        save_eva_files(args, LeCaRDv2_zeroshot_output, 'LeCaRDv2', zeroshot=True)
        save_eva_files(args, LeCaRD_zeroshot_output, 'LeCaRD', zeroshot=True)
    
    if load == True:
        checkpoint_dirs = [i for i in os.listdir(args.training_args['output_dir']) if os.path.isdir(os.path.join(args.training_args['output_dir'], i)) and i.startswith('checkpoint')]
        final_checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
        trainer_state = json.load(open(os.path.join(args.training_args['output_dir'], final_checkpoint_dirs, 'trainer_state.json'), 'r'))
        try:
            model_params = torch.load(os.path.join(trainer_state['best_model_checkpoint'], 'pytorch_model.bin'))
        except:
            from safetensors import safe_open
            model_params = {}
            with safe_open(os.path.join(trainer_state['best_model_checkpoint'], 'model.safetensors'), framework="pt") as f:
                for k in f.keys():
                    model_params[k] = f.get_tensor(k)
        model.load_state_dict(model_params)
    
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

def KELLER_test(args, model, data_provider, load=False):
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
    
    if load == True:
        checkpoint_dirs = [i for i in os.listdir(args.training_args['output_dir']) if os.path.isdir(os.path.join(args.training_args['output_dir'], i)) and i.startswith('checkpoint')]
        final_checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))[-1]
        trainer_state = json.load(open(os.path.join(args.training_args['output_dir'], final_checkpoint_dirs, 'trainer_state.json'), 'r'))
        try:
            model_params = torch.load(os.path.join(trainer_state['best_model_checkpoint'], 'pytorch_model.bin'))
        except:
            from safetensors import safe_open
            model_params = {}
            with safe_open(os.path.join(trainer_state['best_model_checkpoint'], 'model.safetensors'), framework="pt") as f:
                for k in f.keys():
                    model_params[k] = f.get_tensor(k)
        model.load_state_dict(model_params)
    
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