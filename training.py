from transformer import create_mask
import torch
from tqdm.auto import tqdm
import gc

from tokenization import *
from customdataset import *
from transformer import *
from utils import *

def train(model, iterator, optimizer, loss_fn, args, epoch):
    printsave(args.result_path, "====== start training =======")
    model.train()
    epoch_loss = 0
    batch_loss = 0

    scheduler = get_scheduler(optimizer, iterator, args)
    bar = tqdm(enumerate(iterator), total=len(iterator), desc=f'train:{epoch}/{args.epoch}')

    for i, batch in bar:
        src = batch[0].T.to(args.device)
        tgt = batch[1].T.to(args.device)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1,:], args.device)

        optimizer.zero_grad() 
        output = model(src, tgt[:-1,:],
                       src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_output = tgt[1:, :].reshape(-1)  # ignore for target's 
        output = output.reshape(-1, output.shape[-1])
        loss = loss_fn(output, tgt_output)

        loss.backward()     
        optimizer.step() 
        scheduler.step()     
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # gradient clipping

        epoch_loss += loss.item()
        batch_loss += loss.item()

        if (i+1) % 1000 == 0:
            print(f'loss: {batch_loss / 1000}')
            batch_loss = 0

            # gc.collect()
            # torch.cuda.empty_cache()
    # model save every epoch
    torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_{epoch}.pt'))
                                       # update parameters
    printsave(args.result_path, f'epoch: {epoch}')
    printsave(args.result_path, f'epoch loss: {epoch_loss / len(iterator)}')
    # return epoch_loss / len(iterator)


def test(iterator, args, epoch):
    printsave(args.result_path, "====== start testing =======")
    file_path = os.path.join(args.model_dir, args.tokenizer, f'pred_{epoch}.txt')

    model = Transformer_(args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, f'model_{epoch}.pt')))
    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
        bar = tqdm(enumerate(iterator), total=len(iterator), desc=f'test:{epoch}/{args.epoch}')

        for i, batch in bar:
            src = batch[0].T.to(args.device)
            num_tokens = src.shape[0]
            max_len = num_tokens + 5

            unfinished_sequences = torch.ones(1,args.batch_size).to(args.device)
            ys = torch.ones(1, args.batch_size).fill_(0).type(torch.long).to(args.device)
            src_mask, _, src_padding_mask, _ = create_mask(src, ys, args.device)
            context = model.encode(src, src_mask, src_padding_mask).to(args.device) # n, batch, dim
            
            for i in range(max_len-1):
                tgt_mask = (generate_square_subsequent_mask(ys.size(0), args.device)
                    .type(torch.bool))
                out = model.decode(ys, context, tgt_mask)
                out = out.transpose(0, 1)
                prob = model.fc_out(out[:, -1])
                _, next_word = torch.max(prob, dim = 1)

                next_word = next_word * unfinished_sequences + (1-unfinished_sequences) * 2
                ys = torch.cat([ys, next_word], dim=0)
                ys = ys.type(torch.long)

                unfinished_sequences = unfinished_sequences.mul((next_word != 1).long())
                
                if unfinished_sequences.max() == 0:
                    break
            ys = ys.T
            decoded_ys = [' '.join([args.tokenizer.DecodePieces(i) for i in ys[j].tolist() if i not in [0, 1, 2]]) for j in range(args.batch_size)]
            save(file_path, '\n'.join(decoded_ys))
            # for j in range(args.batch_size):
                
            #     s = ' '.join([args.tokenizer.DecodePieces(i) for i in ys[j].tolist() if i not in [0,1,2]])
            #     save(file_path, s)
            # for j in range(args.batch_size):
            #     print(' '.join([tokenizer.DecodePieces(i) for i in ys[:,j] if i not in [0,1,2]]))

 

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
