from torch.nn import Transformer
from torch import nn
import torch
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, args, device):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, args.hidden_dim)
        self.scale      = torch.sqrt(torch.FloatTensor([args.hidden_dim])).to(device)

    def forward(self, src):
        return self.embedding(src) * self.scale
    
class PositionalEncoding(nn.Module):
    def __init__(self, args, max_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(args.dropout_ratio)

        pos = torch.arange(max_length).unsqueeze(1)
        den = torch.exp(torch.arange(0, args.hidden_dim, 2) * (-math.log(10000) / args.hidden_dim))

        pos_embedding = torch.zeros(max_length, 1, args.hidden_dim)
        pos_embedding[:, 0, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 0, 1::2] = torch.cos(pos * den)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # x: [seq_len, batch_size, hidden_dim]
        token_embedding += self.pos_embedding[:token_embedding.size(0), :]
        return self.dropout(token_embedding)
    
    
class Transformer_(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.src_token_embedding = TokenEmbedding(args.input_dim, args, args.device)
        self.tgt_token_embedding = TokenEmbedding(args.output_dim, args, args.device)
        self.positional_encoding = PositionalEncoding(args)

        self.transformer = Transformer(d_model=args.hidden_dim,
                                       nhead=args.n_heads,
                                       num_encoder_layers=args.n_layers,
                                       num_decoder_layers=args.n_layers,
                                       dim_feedforward=args.pf_dim,
                                       dropout=args.dropout_ratio)
        self.fc_out = nn.Linear(args.hidden_dim, args.output_dim)
        
    def forward(self, src, tgt, src_mask, tgt_mask,
                src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.src_token_embedding(src)
        tgt_emb = self.tgt_token_embedding(tgt)

        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)

        output = self.transformer(src_emb, tgt_emb,
                                  src_mask, tgt_mask,
                                  None,  # None for memory_mask
                                  src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.fc_out(output)

    
    def encode(self, src, src_mask, src_padding_mask=None):
        src_emb = self.src_token_embedding(src)
        src_emb = self.positional_encoding(src_emb)
        return self.transformer.encoder(src_emb, src_mask, src_padding_mask)

    
    def decode(self, tgt, memory, tgt_mask):
        tgt_emb = self.tgt_token_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        return self.transformer.decoder(tgt_emb, memory, tgt_mask)
    
    
    
def generate_square_subsequent_mask(size, device):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask==0, float('-inf'))\
                     .masked_fill(mask==1, float(0.0))
    return mask


def create_mask(src, tgt, device):
    src_seq_len = src.shape[0]
    src_mask    = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    tgt_seq_len = tgt.shape[0]
    tgt_mask    = generate_square_subsequent_mask(tgt_seq_len, device)

    pad_idx = 3  # Preprocessor's pad_token_id
    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def initialize_weights(model):
    for p in model.parameters():
        if p.dim() > 1: nn.init.xavier_uniform_(p)
            
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class GreedyTextGenerator:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate_text(self, src, max_length=50):
        src_tensor = src.unsqueeze(0).to(self.device)
        src_mask = (torch.zeros(src_tensor.shape[0], src_tensor.shape[0])).type(torch.bool).to(self.device)
        tgt_tokens = torch.tensor([0], dtype=torch.long).unsqueeze(0).to(self.device)
        print(src_tensor.shape)
        print(src_tensor)
        print(src_mask.shape)
        print(src_mask)

        with torch.no_grad():
            for i in range(max_length):
                tgt_mask = generate_square_subsequent_mask(tgt_tokens.size(1), self.device).type(torch.bool).to(self.device)
                context = self.model.encode(src_tensor, src_mask)
                print(context.shape)
                print(context)
                print(tgt_mask.shape)
                print(tgt_mask)
                print(tgt_tokens.shape)
                print(tgt_tokens)
                output = self.model.decode(tgt_tokens, context, tgt_mask)

                next_token = torch.argmax(output[:, -1, :], dim=-1)
                tgt_tokens = torch.cat((tgt_tokens, next_token.unsqueeze(0)), dim=1)

                if next_token.item() == 1:
                    break

        generated_text = self.tokenizer.decode(tgt_tokens[0].tolist()[1:])  # Skip the <s> token
        return generated_text
 