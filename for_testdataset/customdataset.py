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
        # if args.tokenizer == None:
        #     dataset = [(self.tokenizer.encode(src), self.tokenizer.encode(tgt))
        #                for src, tgt in zip(self.src_lst, self.tgt_lst)
        #                if len(src) > 0 and len(tgt) > 0]
        # # elif args.tokenizer == 'BPE': 
        # #     dataset = [(self.tokenizer.encode_as_ids(src), self.tokenizer.encode_as_ids(tgt))
        # #                for src, tgt in zip(self.src_lst, self.tgt_lst)
        # #                if len(src) > 0 and len(tgt) > 0]
        # else: raise Error('Wrong tokenizer detected...')
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
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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
    return src_sent.to(device), tgt_sent.to(device)


def batch_sampling(sequence_lengths, batch_size):
    """ Composing batch to match similar lengths of sequences """
    seq_lens = [(i, seq_len, tgt_len) for i, (seq_len, tgt_len) in enumerate(sequence_lengths)]
    seq_lens = sorted(seq_lens, key=lambda x: x[1])
    seq_lens = [sample[0] for sample in seq_lens]

    sample_indices = [seq_lens[i:i+batch_size] for i in range(0, len(seq_lens), batch_size)]
    random.shuffle(sample_indices)
    return sample_indices


def make_loader(dataset, batch_size):
    """ Make dataloader using sequence padding & composing batch """
    sequence_lengths = list(map(lambda x: (len(x[0]), len(x[1])), dataset))
    batch_sampler = batch_sampling(sequence_lengths, batch_size)
    loader = DataLoader(dataset,
                        collate_fn=collate_fn,
                        batch_sampler=batch_sampler)
    return loader


def make_all_loaders(args, src1, tgt1, src2, tgt2, src3, tgt3):
    """ Make train, valid, test dataloaders """
    tokenizer = get_tokenizer(args)
    # if args.tokenizer == 'BPE':
        #tokenizer = BPETokenizer(args.ori_tokens_dir, args.noi_tokens_dir)
        #tokenizer = HFTokenizer(args)
        # tokenizer = SPTokenizer(args)
    # else:
    #     tokenizer = Tokenizer()
    train_dataset = CustomDataset(args, src1, tgt1, tokenizer)
    valid_dataset = CustomDataset(args, src2, tgt2, tokenizer)
    test_dataset  = CustomDataset(args, src3, tgt3, tokenizer)
    
    train_loader = make_loader(train_dataset, args.batch_size)
    valid_loader = make_loader(valid_dataset, args.batch_size)
    test_loader  = make_loader(test_dataset, args.batch_size)
    return train_loader, valid_loader, test_loader


# def pre_data(args):


#     return train_loader, valid_loader, test_loader
