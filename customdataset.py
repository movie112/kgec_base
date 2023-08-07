from tokenization import *
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm.auto import tqdm

class CustomDataset(Dataset):
    def __init__(self, args, src_lst, tgt_lst, tokenizer):
        self.src_lst = src_lst
        self.tgt_lst = tgt_lst
        assert len(self.src_lst) == len(self.tgt_lst)

        self.tokenizer = tokenizer
        self.dataset   = self.make_dataset(args)
        
    def make_dataset(self, args):
        # char encoding 시, 앞에 공백 붙는 문제 해결
        if args.tokenizer_name == 'char' or args.tokenizer_name == 'char_c': # encoding 시 앞에 4가 붙는 문제 해결
            dataset = []
            for i in range(len(self.src_lst)):
                src = self.tokenizer.encode(self.src_lst[i])
                tgt = self.tokenizer.encode(self.tgt_lst[i])
                if src[0] == 4: src = src[1:]
                if tgt[0] == 4: tgt = tgt[1:]
                dataset.append((src, tgt))
        else:
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
    src_sent, tgt_sent = [], []
    for src, tgt in batch_samples:
        src_sent.append(torch.tensor([0] + src + [1]))  # bos, eos
        tgt_sent.append(torch.tensor([0] + tgt + [1]))  # bos, eos
    
    src_sent = pad_sequence(src_sent, batch_first=True, padding_value=2)
    tgt_sent = pad_sequence(tgt_sent, batch_first=True, padding_value=2)

    return src_sent, tgt_sent

def make_loader(args, mode):
    data_path = os.path.join(args.data_dir, f"total_resorted_29_{mode}.json")
    print(f'Loading data..."{data_path}')
    with open(data_path, 'r') as f:
        raw_dataset = json.load(f)
    tgt = [item[0] for item in tqdm(raw_dataset)]
    src = [item[1] for item in tqdm(raw_dataset)]

    if mode == 'test':
        # if len(tgt)%args.batch_size != 0:
        #     rest = len(tgt)%args.batch_size
        #     tgt = tgt[:-rest]
        #     src = src[:-rest]

        # gold.txt saving
        gold_path = os.path.join(args.result_dir, args.tokenizer_name, f'gold.txt')
        if not os.path.exists(gold_path):
            print('saving gold data...')
            with open(gold_path, 'w', encoding='utf-8') as f:
                f.write('S ')
                f.write('\nS '.join(tgt))
            print('finish saving gold data...')
    # loader make
    print(f'making {mode} loader...')
    dataset = CustomDataset(args, src, tgt, args.tokenizer)
    # curriculum X
    if mode == 'train' and args.tokenizer_name == 'bpe' and args.tokenizer_name == 'char':
        print('making random sampler...')
        loader = DataLoader(dataset,
                              batch_size=args.batch_size, 
                              sampler = RandomSampler(dataset), 
                              collate_fn=collate_fn,
                              num_workers=2)
    else: # curriculum O
        print('making sequential sampler...')
        loader = DataLoader(dataset,
                              batch_size=args.batch_size if mode != 'test' else 1, # inference, batch==1
                              sampler = SequentialSampler(dataset), 
                              collate_fn=collate_fn, 
                              num_workers=2)
    return loader
