from tokenization import *
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, args, src_lst, tgt_lst, tokenizer):
        self.src_lst = src_lst
        self.tgt_lst = tgt_lst
        assert len(self.src_lst) == len(self.tgt_lst)

        self.tokenizer = tokenizer
        self.dataset   = self.make_dataset(args)
        
    def make_dataset(self, args):
        if args.tokenizer_name == 'char': # encoding 시 앞에 4가 붙는 문제 해결
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
        print(self.src_lst[0])
        print(dataset[0])
        return dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
def collate_fn(batch_samples):
    """ Sequence padding in batch """
    # pad_token_id == 2 
    src_sent, tgt_sent = [], []

    for src, tgt in batch_samples:
        src_sent.append(torch.tensor([0] + src + [1]))  # bos, eos
        tgt_sent.append(torch.tensor([0] + tgt + [1]))  # bos, eos
    
    src_sent = pad_sequence(src_sent, batch_first=True, padding_value=2)
    tgt_sent = pad_sequence(tgt_sent, batch_first=True, padding_value=2)

    # for i in range(len(batch_samples)):
    #     batch_samples[i][1].insert(0, 0) # bos
    #     batch_samples[i][1].append(1) # eos
    
    # src_sent = pad_sequence([torch.tensor(src) for src, _ in batch_samples],
    #                          batch_first=True,
    #                          padding_value=2)
    # tgt_sent = pad_sequence([torch.tensor(tgt) for _, tgt in batch_samples],
    #                          batch_first=True,
    #                          padding_value=2)
    return src_sent, tgt_sent

def make_all_loaders(args, epoch):
    # load data
    print(f'Loading data..."{args.data_path}"')
    with open(args.data_path, 'r') as f:
        raw_dataset = json.load(f)

    # for testing !!! 
    # raw_dataset = raw_dataset[:1000000]
    # print('test indices...')
    # test_indices = [0 + i*10 for i in tqdm(range(1000000//10))]

    print('test indices...')
    test_indices = [0 + i*10 for i in tqdm(range(6036434//10))]

    print('splitting data...')
    df = pd.DataFrame(raw_dataset)
    # Split the dataset into two parts based on test_indices using boolean indexing
    train_mask = ~df.index.isin(test_indices)
    train_src = df.loc[train_mask, 1].tolist()
    train_tgt = df.loc[train_mask, 0].tolist()
    test_src = df.loc[test_indices, 1].tolist()
    test_tgt = df.loc[test_indices, 0].tolist()

    print('saving gold data...')
    rest = len(test_tgt)%args.batch_size
    test_tgt = test_tgt[:-rest]
    test_src = test_src[:-rest]

    with open(os.path.join(args.result_dir, args.tokenizer_name, f'gold_{epoch}.txt'), 'w') as f:
        f.write('S ')
        f.write('\nS '.join(test_tgt))
    print('finish saving gold data...')
    
    train_dataset = CustomDataset(args, train_src, train_tgt, args.tokenizer)
    test_dataset  = CustomDataset(args, test_src, test_tgt, args.tokenizer)


    print('making train loaders...')
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,  
                              collate_fn=collate_fn)
    print('making test loaders...')
    test_loader  = DataLoader(test_dataset,
                                batch_size=args.batch_size, ##
                                collate_fn=collate_fn)
    return train_loader, test_loader

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
