from tokenization import *
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch
import random


class CustomDataset(Dataset):
    def __init__(self, args, src_lst, tgt_lst, tokenizer):
        self.src_lst = src_lst
        self.tgt_lst = tgt_lst
        assert len(self.src_lst) == len(self.tgt_lst)

        self.tokenizer = tokenizer
        self.dataset   = self.make_dataset(args)

        
    def make_dataset(self, args):
        dataset = [(self.tokenizer.encode(src), self.tokenizer.encode(tgt))
                       for src, tgt in zip(self.src_lst, self.tgt_lst)
                       if len(src) > 0 and len(tgt) > 0]
        return dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
def collate_fn(batch_samples):
    """ Sequence padding in batch """
    pad_token_id = 2  # pad_token_id == 2 
    for i in range(len(batch_samples)):
        batch_samples[i][1].insert(0, 0) # bos
        batch_samples[i][1].append(1) # eos
    
    src_sent = pad_sequence([torch.tensor(src) for src, _ in batch_samples],
                             batch_first=True,
                             padding_value=pad_token_id)
    tgt_sent = pad_sequence([torch.tensor(tgt) for _, tgt in batch_samples],
                             batch_first=True,
                             padding_value=pad_token_id)
    return src_sent, tgt_sent

def make_all_loaders(args, epoch):
    # load data
    print(f'Loading data...{args.data_path}')
    with open(args.data_path, 'r') as f:
        raw_dataset = json.load(f)
    test_indeices = [0 + i*10 for i in range(16550304//10)]

    train_src, train_tgt, test_src, test_tgt = [], [], [], []
    for i in range(len(raw_dataset)):
        if i not in test_indeices:
            train_src.append(raw_dataset[i][1])
            train_tgt.append(raw_dataset[i][0])
        else:
            test_src.append(raw_dataset[i][1])
            test_tgt.append(raw_dataset[i][0])
            
    print('saving gold data...')
    with open(os.path.join(args.result_path, args.tokenizer, f'gold_{epoch}.txt'), 'w') as f:
        f.write('\n'.join(test_tgt) + '\n')
        
    train_dataset = CustomDataset(args, train_src, train_tgt, args.tokenizer)
    test_dataset  = CustomDataset(args, test_src, test_tgt, args.tokenizer)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,  
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,
                                batch_size=args.batch_size, ##
                                collate_fn=collate_fn)
    return train_loader, test_loader

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask