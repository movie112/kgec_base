import torch
from tqdm.auto import tqdm

from tokenization import *
from customdataset import *
from transformer import *
from utils import *

def training(args):
    
    model = Transformer_(args)
    model.apply(initialize_weights)
    #
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
    best_epoch = 0
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
                print('')
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
                    print('')
                    printsave(args.result_path, f'loss: {batch_loss / 1000}')
                    batch_loss = 0
            printsave(args.result_path, f'total loss: {total_loss / len(valid_loader)}')
            if best_valid_loss > total_loss / len(valid_loader):
                stop_cnt = 0
                best_valid_loss = total_loss / len(valid_loader)
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.tokenizer_name, 'model.pt'))
                printsave(args.result_path, 'saving model...')
                best_epoch = epoch
            else:
                stop_cnt += 1
                if stop_cnt == args.patience:
                    printsave(args.result_path, 'early stopping')
                    break
                printsave('stop_cnt:', stop_cnt)    
            if epoch == 9:
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.tokenizer_name, 'model_10.pt'))
            elif epoch == 14:
                torch.save(model.state_dict(), os.path.join(args.model_dir, args.tokenizer_name, 'model_15.pt'))
    print('best epoch:', best_epoch)

def translation(args):
    printsave(args.result_path, "====== start testing =======")
    test_loader = make_loader(args, 'test')
    file_path = os.path.join(args.result_dir, args.tokenizer_name, f'pred.txt')

    model = Transformer_(args)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.tokenizer_name, f'model.pt')))
    model = model.to(args.device)
    model.eval()

    with torch.no_grad():
        tgt_lst = []
        bar = tqdm(enumerate(test_loader), total=len(test_loader))
        for i, batch in bar:
            src = batch[0].T.to(args.device)  
            num_tokens = src.shape[0]
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
            tgt_tokens = beam_search_decode(
                model,  src, src_mask, args, max_len=num_tokens + 5, beam_size=3, start_symbol=0)
            tgt_tokens = tgt_tokens.tolist()
            tgt_lst.append(args.tokenizer.decode(tgt_tokens))
            # print('translated')
        with open(file_path, 'w') as f:
            f.write('\n'.join(tgt_lst))


def beam_search_decode(model, src, src_mask, args, max_len, beam_size, start_symbol):
    src = src.to(args.device)
    src_mask = src_mask.to(args.device)

    memory = model.encode(src, src_mask)

    # 시작 토큰 추가
    ys = torch.tensor([start_symbol], dtype=torch.long).to(args.device)

    # 빈 beam list 생성
    beam_list = [(0.0, ys, False)]  # (log probability, sequence)

    for i in range(max_len - 1):
        new_beam_list = []

        for log_prob, sequence, finish in beam_list:
            tgt_mask = generate_square_subsequent_mask(sequence.size(0), args.device).type(torch.bool).to(args.device)
            out = model.decode(sequence.unsqueeze(1), memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.fc_out(out[:, -1])
            log_probs, next_words = torch.topk(F.log_softmax(prob, dim=1), beam_size, dim=1)

            for j in range(beam_size):
                new_sequence = torch.cat([sequence, next_words[0][j].unsqueeze(0)], dim=0)
                new_log_prob = log_prob + log_probs[0][j].item()

                if finish: # 끝났으면
                    new_beam_list.append((log_prob, sequence, True))
                else:
                    if next_words[0][j].item() == 1: # EOS 토큰일 때, 결과 리스트에 추가
                        new_beam_list.append((log_prob, sequence, True))
                    else:# EOS 토큰이 아니면 다시 beam list에 추가
                        new_beam_list.append((new_log_prob, new_sequence, False))

        # beam_size에 따라 점수에 따라 정렬하고 상위 beam_size개를 선택
        new_beam_list.sort(key=lambda x: x[0], reverse=True)
        beam_list = new_beam_list[:beam_size]
