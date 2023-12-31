from training import *
from transformer import *
from customdataset import *
from tokenization import *
from utils import printsave

import os
import torch.optim as optim
from torch import nn
import torch

import pandas as pd
import argparse
import time
from tqdm.auto import tqdm



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type    = str,
        default = '',
        help    = 'original data file must be located in this directory'
    )
    parser.add_argument(
        '--model_dir',
        type    = str,
        default = '',
        help    = 'trained model will be save in this directory'
    )
    parser.add_argument(
        '--result_dir',
        type    = str,
        default = '',
        help    = 'training result will be save in this directory'
    )
    parser.add_argument(
        '--epochs',
        type    = int,
        default = 10,
        help    = 'training epochs' 
    )
    parser.add_argument(
        '--batch_size',
        type    = int,
        default = 64,
        help    = 'batch size when training' 
    )
    parser.add_argument(
        '--vocab_size',
        type    = int,
        default = 96,
        help    = 'vocabulary size used in custom tokenizer, default value for phoneme-level tokenization'
    )
    parser.add_argument(
        '--input_dim',
        type    = int,
        default = 96,
        help    = 'encoder input vocab size, default value for phoneme-level tokenization'
    )
    parser.add_argument(
        '--output_dim',
        type    = int,
        default = 96,
        help    = 'decoder input vocab size, default value for phoneme-level tokenization'
    )
    parser.add_argument(
        '--hidden_dim',
        type    = int,
        default = 512,
        help    = 'hidden dimension in Transformer'
    )
    parser.add_argument(
        '--n_heads',
        type    = int,
        default = 8,
        help    = 'number of heads in Transformer'
    )
    parser.add_argument(
        '--n_layers',
        type    = int,
        default = 6,
        help    = 'number of layers in Transformer'
    )
    parser.add_argument(
        '--pf_dim',
        type    = int,
        default = 2048,
        help    = 'feedforward dimension in Transformer'
    )
    parser.add_argument(
        '--dropout_ratio',
        type    = float,
        default = 0.1,
        help    = 'dropout ratio in Transformer'
    )
    parser.add_argument(
        '--lr',
        type    = float,
        default = 0.0001,
        help    = 'learning rate in Adam optimizer'
    )
    parser.add_argument(
        '--beta1',
        type    = float,
        default = 0.9,
        help    = 'beta1 in Adam optimizer'
    )
    parser.add_argument(
        '--beta2',
        type    = float,
        default = 0.98,
        help    = 'beta2 in Adam optimizer'
    )
    parser.add_argument(
        '--eps',
        type    = float,
        default = 1e-9,
        help    = 'eps in Adam optimizer'
    )
    parser.add_argument(
        '--clip',
        type    = int,
        default = 1,
        help    = 'gradient clipping'
    )
    parser.add_argument(
        '--tokenizer',
        type    = str,
        default = None,
        help    = 'tokenizing using phoneme or BPE'
    )
    parser.add_argument(
        '--tokenizer_dir',
        type    = str,
        default = None,
        help    = 'BPE tokenizer must be located in this directory'
    )
    args = parser.parse_args()

    args.tokenizer_name = 'bpe'
    args.tokenizer_dir = '/HDD//kgec/tokenizer'
    args.vocab_model_path = f'/HDD//kgec/tokenizer/{args.tokenizer_name}.model'
    args.data_dir = '/HDD//kgec'

    args.tokenizer_name = 'bpe' # bpe, bpe_c, char, char_c
    args.result_dir = '/HDD//kgec/result'
    args.result_path = f'/HDD//kgec/result/log_{args.tokenizer_name}.txt'
    args.model_dir = '/HDD//kgec/model'
    # args.gold_path = '/HDD//KGEC/result/bpe/gold.txt'
    # args.pred_path = '/HDD//KGEC/result/bpe/pred.txt'

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if not os.path.exists(os.path.join(args.result_dir, args.tokenizer_name)):
        os.makedirs(os.path.join(args.result_dir, args.tokenizer_name))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(os.path.join(args.model_dir, args.tokenizer_name)):
        os.makedirs(os.path.join(args.model_dir, args.tokenizer_name))
    

    args.lr = 5e-6
    args.num_warmup_steps = 2000
    args.batch_size = 64
    args.epoch = 20
    args.dropout_ratio = 0.2
    args.hidden_dim = 768
    args.patience = 4

    cuda_num = 1
    # define device
    args.device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {args.device}\n')

    set_seed(42)
    
    args.tokenizer = get_tokenizer(args)
    args.vocab_size = args.tokenizer.get_piece_size()
    args.input_dim = args.vocab_size
    args.output_dim = args.vocab_size
    print(f'vocab size: {args.vocab_size}\n')


    # # train
    training(args)
    # # test
    translation(args)

    # metric
    scorer_path = '/home//workspace/kgec/scripts/m2scorer.py'
    gold_path = os.path.join(args.result_dir, args.tokenizer_name, f'gold.txt')
    pred_path = os.path.join(args.result_dir, args.tokenizer_name, f'pred.txt')

    os.system(f'{scorer_path} {pred_path} {gold_path}')
