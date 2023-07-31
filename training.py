import torch
from tqdm.auto import tqdm

from tokenization import *
from customdataset import *
from transformer import *
from utils import *

def training(args):
    
    model = Transformer_(args)
    model.apply(initialize_weights)
    model = model.to(args.device)
    print(f'model has {count_parameters(model):,} trainable parameters\n')

    train_loader = make_loader(args, 'train')
    valid_loader = make_loader(args, 'valid')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=(args.beta1, args.beta2),
                                 eps=args.eps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # pad_token_id == 2
    scheduler = get_scheduler(optimizer, train_loader, args)


    best_valid_loss = np.inf
    for epoch in range(args.epoch):
        printsave(args.result_path, f"====== {epoch} =======")
        model.train()
        batch_loss = 0
        bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'train:{epoch}/{args.epoch}')
        for step, batch in bar:
            src = batch[0].T.to(args.device)
            tgt = batch[1].T.to(args.device)
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1,:], args.device)

            optimizer.zero_grad() 
            output = model(src, tgt[:-1,:],
                           src_mask, tgt_mask,
                           src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_output = tgt[1:, :].reshape(-1)
            output = output.reshape(-1, output.shape[-1])
            loss = loss_fn(output, tgt_output)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # gradient clipping
            optimizer.step()
            scheduler.step()

            batch_loss += loss.item()
            if (step+1) % 2000 == 0:
                printsave(args.result_path, f'loss: {batch_loss / 2000}')
                batch_loss = 0

        printsave(args.result_path, "====== start validation =======")
        batch_loss = 0
        total_loss = 0
        stop_cnt = 0
        with torch.no_grad():
            model.eval()
            bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'valid:{epoch}/{args.epoch}')
            for step, batch in bar:
                src = batch[0].T.to(args.device)
                tgt = batch[1].T.to(args.device)
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1,:], args.device)

                output = model(src, tgt[:-1,:],
                               src_mask, tgt_mask,
                               src_padding_mask, tgt_padding_mask, src_padding_mask)

                tgt_output = tgt[1:, :].reshape(-1)
                output = output.reshape(-1, output.shape[-1])
                loss = loss_fn(output, tgt_output)
                batch_loss += loss.item()
                total_loss += loss.item()

                if (step+1) % 1000 == 0:
                    printsave(args.result_path, f'loss: {batch_loss / 1000}')
                    batch_loss = 0
            printsave(args.result_path, f'total loss: {total_loss / len(valid_loader)}')
            if best_valid_loss > total_loss / len(valid_loader):
                stop_cnt = 0
                best_valid_loss = total_loss / len(valid_loader)
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.tokenizer_name, f'model.pt'))
                printsave(args.result_path, 'saved model at', os.path.join(args.model_dir, args.tokenizer_name, f'model.pt'))
            else:
                stop_cnt += 1
                if stop_cnt == args.patience:
                    printsave(args.result_path, 'early stopping')
                    break

def testing(args):
    printsave(args.result_path, "====== start testing =======")
    test_loader = make_loader(args, 'test')
    file_path = os.path.join(args.result_dir, args.tokenizer_name, f'pred.txt')

    model = Transformer_(args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.tokenizer_name, f'model_.pt')))
    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
        ys_lst = []
        bar = tqdm(enumerate(test_loader), total=len(test_loader))

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
            ys = ys.T.tolist()
            ys_lst += ys
        print(f'saving pred.txt at {file_path}')
        decoded_ys = [args.tokenizer.decode(ys) for ys in ys_lst ]
        with open(file_path, 'w') as f:
            f.write('\n'.join(decoded_ys))
    gold_path = os.path.join(args.result_dir, args.tokenizer_name, f'gold.txt')
    pred_path = os.path.join(args.result_dir, args.tokenizer_name, f'pred.txt')

    os.system(f'/home/yeonghwa/workspace/kgec/baseline/scripts/m2scorer.py {pred_path} {gold_path}')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
