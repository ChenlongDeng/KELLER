import argparse
import json
import yaml
import torch
import os
import random
import numpy as np
import logging
from model.Model import Get_Model
from dataset.Dataset import Get_DataProvider
from tools.Train import train
from tools.Test import test
from tools.Metrics import compute_metrics_normal, save_all_results

# Arguments Setting
logging.disable(logging.WARNING)
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default='./config/SAILER.yaml', help='The path of config file')
parser.add_argument('--local_rank', default=0)
parser.add_argument('--hard_negative_num', default=0, type=int)
parser.add_argument('--test_zeroshot', default=False, type=bool)
parser.add_argument('--test_mode', default=False, type=bool)
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    args.__dict__.update(yaml.load(f, Loader=yaml.FullLoader))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])
args.device = torch.device(args.device)
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    # Set a fixed seed
    set_seed(seed=args.training_args['seed'])

    # Prepare Model
    model = Get_Model(args)
    if args.device.type == 'cuda':
        model = model.to(f'cuda:{args.local_rank}')
    
    # Prepare data
    data_provider = Get_DataProvider(args)
        
    if args.test_mode:
        test(args, model, data_provider, load=False)
    else:
        train(args, model, data_provider)

    