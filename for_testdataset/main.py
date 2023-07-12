from training import *
from transformer import *
from customdataset import *
from tokenization import *
from utils import printsave

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
    parser.add_argument(
        '--mode',
        type    = str,
        default = None,
        help    = 'GEC using blank or not'
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
    ##
    parser.add_argument(
        '--vocab_model',
        type    = str,
        default = '',
    )
    args = parser.parse_args()

    args.tokenizer_dir = './tokenizer'
    args.result_dir = './result/result_bpe.txt'
    args.data_path = './data/testdataset.json'
    args.vocab_model = './tokenizer/bpe.model'
    args.model_dir = './model/bpe.pt'
    # args.tokenizer = 'bpe'
    args.vocab_size = 16000
    args.input_dim = 16000
    args.output_dim = 16000
    args.batch_size = 32
    
    # define device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}\n')
    
    with open(args.data_path, 'r') as f:
        raw_dataset = json.load(f)
        pairs = [(d['origin'], d['augmented']) for d in raw_dataset['documents']]
        text, labels = list(zip(*pairs))[0], list(zip(*pairs))[1] 
    
    args.data_size = len(text)
    cut = args.data_size
    train_src, train_tgt = text[:int(cut*0.8)], labels[:int(cut*0.8)]
    valid_src, valid_tgt = text[int(cut*0.8):int(cut*0.9)], labels[int(cut*0.8):int(cut*0.9)]
    test_src, test_tgt = text[int(cut*0.9):], labels[int(cut*0.9):]
    
    train_loader, valid_loader, test_loader = make_all_loaders(args,
                                                               train_src, train_tgt,
                                                               valid_src, valid_tgt,
                                                               test_src, test_tgt)
    print("finish making loaders") 


    # define model
    model = Transformer_(args, device)
    model.apply(initialize_weights)
    model = model.to(device)
    print(f'model has {count_parameters(model):,} trainable parameters\n')
    
    
    # define optimizer & loss func
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.eps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # pad_token_id == 2
    
    
    # train
    best_valid_loss = float('inf')
    file = open(args.result_dir, 'w')

    for epoch in range(args.epochs):
        print('===== EPOCH {} ====='.format(epoch+1))
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, loss_fn, args.clip, device)
        print(train_loss)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), args.model_dir)
            
        # print(f'\tEpoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        # print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
        # print('\n')
        printsave(file, f'\tEpoch: {epoch + 1:02} ')#| Time: {epoch_mins}m {epoch_secs}s')
        printsave(file, f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
        printsave(file, f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
        printsave(file, '\n')


    # test
    test_loss = evaluate(model, test_loader, loss_fn, device)
    print(f'\tTest Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')
    file.close()