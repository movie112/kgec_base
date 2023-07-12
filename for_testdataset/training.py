from transformer import create_mask
import torch
from tqdm.auto import tqdm
import gc



def train(model, iterator, optimizer, loss_fn, clip, device):
    model.train()
    epoch_loss = 0
    bar = tqdm(enumerate(iterator), total=len(iterator))

    for i, batch in bar:
        src = batch[0].T
        tgt = batch[1].T

        optimizer.zero_grad()  # make gradients zero before backpropagation

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1,:], device)

        output = model(src, tgt[:-1,:],
                       src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_output = tgt[1:, :].reshape(-1)  # ignore for target's 
        output = output.reshape(-1, output.shape[-1])

        loss = loss_fn(output, tgt_output)

        loss.backward()     
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # gradient clipping
        optimizer.step()           
                                              # compute gradient
        epoch_loss += loss.item()

        # if i % 300 == 0:
        #     # print(f'step: {i} loss: {loss.item()}')
            
        #     gc.collect()
        #     torch.cuda.empty_cache()

                                       # update parameters
    return epoch_loss / len(iterator)


def evaluate(model, iterator, loss_fn, device):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        bar = tqdm(enumerate(iterator), total=len(iterator))
        for i, batch in bar:
            src = batch[0].T
            tgt = batch[1].T

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt[:-1,:], device)

            output = model(src, tgt[:-1, :],
                        src_mask, tgt_mask,
                        src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt = tgt[1:, :].reshape(-1)  # ignore for target's 
            output = output.reshape(-1, output.shape[-1])

            loss = loss_fn(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs