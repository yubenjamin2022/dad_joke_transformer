import torch
from torch import nn
from einops import rearrange, repeat
from torch.utils.data import Dataset
import os
import math


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.num_heads = num_heads
        self.scale = dim_head ** -0.5 # scaling factor for attention mechanism

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = - 1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False) # (batch, seq_len, dim) - > (batch, seq_len, inner_dim * 3)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        x = self.norm(x) # (batch, seq_len, dim)
        qkv = self.to_qkv(x).chunk(3, dim = -1) # (batch, seq_len, dim) -> (batch, seq_len, inner_dim) * 3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) # (batch, seq_len, inner_dim) -> (batch, seq_len, num heads, size_head)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale # attention calculation (batch, seq_len, seq_len)

        # masking elements, diagonal excluded

        mask = torch.ones(dots.shape[-2], dots.shape[-1]) # (seq_len, seq_len)
        mask = torch.triu(mask, diagonal = 1) # include diagonals in the zeros
        mask = mask.to(dots.device)

        mask = dots.masked_fill(mask == 1, float('-inf')) # masking operation 

        attention = self.attend(dots) # softmax

        out = torch.matmul(attention, v)
        out = rearrange(out, 'b h n d - > b n (h d)') # (batch, seq_len, num heads, size of head) - > (batch, seq_len, inner_dim)

        return self.to_out(out) # final linear layer


class Transformer(nn.Module):
    def __init__(self, vocab_size, dim, depth, num_heads, dim_head, mlp_dim, seq_len):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, dim)

        def get_positional_encoding(seq_len, dim):
            position = torch.arange(seq_len).unsqueeze(1) # (seq_len) -> (seq_len, 1)
            div_term = torch.exp(torch.arange(0, dim, 2) * -math.log(10000.0) / dim) # -1*exp(ln(10000)*2i/dim), i being element in dim
            pe = torch.zeros(seq_len, dim)
            pe[:, 0::2] = torch.sin(position * div_term) # even terms
            pe[:, 1::2] = torch.cos(position * div_term) # odd terms
            return pe.unsqueeze(0) # (1, seq_len, dim)
        
        self.PE = nn.Parameter(get_positional_encoding(seq_len, dim), requires_grad = False)

        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.output = nn.Linear(dim, vocab_size) # classification into words
        self.seq_len = seq_len
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, num_heads=num_heads, dim_head = dim_head),
                FeedForward(dim = dim, hidden_dim = mlp_dim)
            ]))

    def forward(self, x):
        b, t, f = x.shape # batch, seq_len, token dim
        x = x + self.PE[:, :t, :] # adding positional encodings
        for attention, ff in self.layers:
            x = attention(x)
            x = ff(x)
        x = self.norm(x)
        x = self.output(x)
        return x

    def embed(self, x):
        return self.embed(x)
    
class TextCSVDatset(Dataset):
    def __init__(self, text_arr, seq_len, tokenizer):
        self.tokenizer = tokenizer
        tokens = [tokenizer(text)["input_ids"] for text in text_arr] # tokenize sequences
        tokens = [[tokenizer.bos_token_id] + item + [tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (seq_len +1 - len(item)) for item in tokens] # add BoS and EoS and padding tokens
        tokens = [item[:seq_len+1] for item in tokens] # truncate sequences
        self.tokens = torch.tensor(tokens)

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        input_seq = self.tokens[idx][:-1]
        target_seq = self.tokens[idx][1:]
        return input_seq, target_seq
    


