import math

import torch
from einops import rearrange, repeat
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MHSA(nn.Module):
    def __init__(self, dims, num_heads=8, dropout=0.0):
        super().__init__()
        assert dims % num_heads == 0, "dim must be divided into heads"

        self.num_heads = num_heads
        self.scale = (dims // num_heads) ** -0.5

        self.norm = nn.LayerNorm(dims)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dims, dims * 3, bias=False)
        self.to_out = nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class MHCA(nn.Module):
    def __init__(self, query_dims, context_dims, out_dims, num_heads=8, dropout=0.1):
        super().__init__()
        assert out_dims % num_heads == 0, "dim must be divided into heads"

        self.num_heads = num_heads
        self.scale = (out_dims // num_heads) ** -0.5

        # Add layer normalization for query and context
        self.norm_q = nn.LayerNorm(query_dims)
        self.norm_context = nn.LayerNorm(context_dims)

        # Existing projections
        self.to_q = nn.Linear(query_dims, out_dims, bias=False)
        self.to_k = nn.Linear(context_dims, out_dims, bias=False)
        self.to_v = nn.Linear(context_dims, out_dims, bias=False)

        # Add output projection
        self.to_out = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        # Apply layer normalization first
        x = self.norm_q(x)
        context = self.norm_context(context)

        # Project and split heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Rearrange to multiple heads
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # Add dropout to attention weights

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # Apply output projection
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MHSA(dims=dim, num_heads=heads, dropout=dropout),
                        FeedForward(dim, dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class KTAdaptor(nn.Module):
    def __init__(self, n_tasks, in_dims, depths=2, num_heads=4, target_dim=None):
        super().__init__()
        self.transformer = Transformer(in_dims, depths, num_heads)
        self.knowledge_token = nn.Parameter(torch.rand(in_dims))
        self.n_tasks = n_tasks

        pe = torch.zeros(n_tasks, in_dims)
        position = torch.arange(0, n_tasks, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, 1, 2).float() * (-math.log(10000.0) / in_dims)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)
        self.register_buffer("sw", torch.tensor([True]))

        self.mlp = (
            nn.Linear(in_dims, target_dim) if in_dims > target_dim else nn.Identity()
        )

    def forward(self, tokens, control_vec):
        """
        tokens.shape = [b, N, 1024]
        control_vec = [False, False, True, ..., False]
                       -> "True" remains the selected task
                       -> control_vec.shape = N
        """
        b_size = tokens.shape[0]
        facial_token = repeat(
            self.knowledge_token, "d -> b c d", b=b_size, c=1
        ).contiguous()

        tokens = torch.cat([facial_token, tokens], dim=1)
        tokens = tokens + self.pe[: self.n_tasks + 1]
        tokens = self.transformer(tokens)

        control_vec = torch.cat([self.sw, control_vec])
        tokens = tokens[:, control_vec]

        return self.mlp(tokens).mean(dim=1), torch.tensor(0).to(tokens.device)

    def forward_inference(self, token, control_vec):
        """
        token.shape = [b, 1, 1024]
        control_vec = [False, False, True, ..., False]
                       -> "True" remains the selected task
        """

        b_size = token.shape[0]
        facial_token = repeat(
            self.knowledge_token, "d -> b c d", b=b_size, c=1
        ).contiguous()

        tokens = torch.cat([facial_token, token], dim=1)
        control_vec = torch.cat([self.sw, control_vec])
        tokens = tokens + self.pe[: self.n_tasks + 1][control_vec]
        tokens = self.transformer(tokens)

        return self.mlp(tokens).mean(dim=1)
