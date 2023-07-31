import transformers
import os
import torch
import numpy as np

def printsave(save_path, message):
    print(message)
    with open(save_path, 'a') as f:
        f.write(message + '\n')
    return True
def save(save_path, message):
    with open(save_path, 'a') as f:
        f.write(message + '\n')
    return True

def get_scheduler(optimizer, train_dataloader, args, name='linear'):
    if name == 'linear':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    elif name == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=len(train_dataloader)*args.epochs,
            last_epoch=-1
        )
    return scheduler 

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
