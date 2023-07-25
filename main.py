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
    # parser.add_argument(
    #     '--data_size',
    #     type    = int,
    #     default = '',
    #     help    = 'entire data size'
    # )
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
    # parser.add_argument(
    #     '--mode',
    #     type    = str,
    #     default = None,
    #     help    = 'GEC using blank or not'
    # )

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
    #
    # parser.add_argument(
    #     '--vocab_model',
    #     type    = str,
    #     default = '',
    # )
    args = parser.parse_args()

    args.tokenizer_name = 'bpe'
    args.tokenizer_dir = './HDD//kgec/tokenizer'
    args.vocab_model_path = f'/HDD//kgec/tokenizer/{args.tokenizer_name}.model'
    args.data_dir = '/HDD//KGEC'

    args.result_dir = '/HDD//kgec/result'
    args.result_path = f'/HDD//kgec/result/log_{args.tokenizer_name}.txt'
    args.model_dir = '/HDD//kgec/model'
    # args.gold_path = '/HDD//KGEC/result/bpe/gold.txt'
    # args.pred_path = '/HDD//KGEC/result/bpe/pred.txt'

    args.lr = 1e-5
    args.num_warmup_steps = 1000
    args.batch_size = 64
    args.epoch = 30

    cuda_num = 3
    # define device
    args.device = torch.device(f'cuda:{cuda_num}' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {args.device}\n')

    args.tokenizer = get_tokenizer(args)
    args.vocab_size = args.tokenizer.get_piece_size()
    args.input_dim = args.vocab_size
    args.output_dim = args.vocab_size
    print(f'vocab size: {args.vocab_size}\n')

    # define model 
    model = Transformer_(args)
    model.apply(initialize_weights)
    model = model.to(args.device)
    print(f'model has {count_parameters(model):,} trainable parameters\n')
    
    # define optimizer & loss func
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.eps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # pad_token_id == 2

    for epoch in range(args.epoch):
        args.data_path = os.path.join(args.data_dir, f'total_cut_{epoch+9}.json')
        train_loader, test_loader = make_all_loaders(args, epoch)
        print("finish making loaders")

        print('test_loader', len(test_loader)) 
    
    # # #     # # training 
        train(model, train_loader, optimizer, loss_fn, args, epoch)
        del train_loader

    #     # test
        test(test_loader, args, epoch)

        gold_path = os.path.join(args.result_dir, args.tokenizer_name, f'gold_{epoch}.txt')
        pred_path = os.path.join(args.result_dir, args.tokenizer_name, f'pred_{epoch}.txt')

        os.system(f'/home//workspace/kgec/scripts/m2scorer.py {pred_path} {gold_path}')
        os.path.join(args.result_dir, args.tokenizer_name, f'gold_{epoch}.txt')
        os.path.join(args.result_dir, args.tokenizer_name, f'pred_{epoch}.txt')
        
