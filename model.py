import torch
from torch import nn


class FeedForwardModule(nn.Module):
    def __init__(self, dim, dim_multiplier, dropout=0):
        super().__init__()

        self.module = nn.Sequential(
            nn.Linear(in_features=dim, out_features=int(dim * dim_multiplier)),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(in_features=int(dim * dim_multiplier), out_features=dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.module(x)


class AttentionModule(nn.Module):
    def __init__(self, dim, num_heads, dropout=0, attn_dropout=0):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError('Dimension mismatch: hidden_dim should be a multiple of num_heads.')

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1 / self.head_dim ** 0.5

        self.attn_dropout = attn_dropout

        self.query_linear = nn.Linear(in_features=dim, out_features=dim)
        self.key_linear = nn.Linear(in_features=dim, out_features=dim)
        self.value_linear = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, attn_mask):
        batch_size, length, dim = x.shape

        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)

        queries = queries.reshape(batch_size, length, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, length, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, length, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = self.scale * (queries @ keys.mT)
        attn_scores += attn_mask

        if self.attn_dropout > 0:
            dropout_probs = self.attn_dropout * torch.ones_like(attn_scores)
            dropout_mask = torch.bernoulli(dropout_probs).to(bool)
            attn_scores[dropout_mask] = - torch.inf

        attn_probs = torch.softmax(attn_scores, dim=-1)

        x = attn_probs @ values
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, length, dim)

        x = self.output_linear(x)
        x = self.dropout(x)

        return x


class TransformerModule(nn.Module):
    def __init__(self, dim, num_heads, dim_multiplier, dropout, attn_dropout):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(dim)
        self.attention_module = AttentionModule(dim=dim, num_heads=num_heads, dropout=dropout,
                                                attn_dropout=attn_dropout)
        self.layernorm_2 = nn.LayerNorm(dim)
        self.feedforward_module = FeedForwardModule(dim=dim, dim_multiplier=dim_multiplier, dropout=dropout)

    def forward(self, x, attn_mask):
        x = x + self.attention_module(self.layernorm_1(x), attn_mask)
        x = x + self.feedforward_module(self.layernorm_2(x))

        return x


class Model(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, hidden_dim_multiplier, num_token_embeddings,
                 num_pos_embeddings, dropout, attn_dropout):

        super().__init__()

        self.token_embeddings = nn.Embedding(num_embeddings=num_token_embeddings, embedding_dim=hidden_dim)
        self.pos_embeddings = nn.Embedding(num_embeddings=num_pos_embeddings, embedding_dim=hidden_dim)

        self.transformer_modules = nn.ModuleList(
            TransformerModule(dim=hidden_dim, num_heads=num_heads, dim_multiplier=hidden_dim_multiplier,
                              dropout=dropout, attn_dropout=attn_dropout)
            for _ in range(num_layers)
        )

        self.output_layernorm = nn.LayerNorm(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=num_token_embeddings, bias=False)

    def forward(self, inputs, attn_mask):
        device = inputs.device

        x = self.token_embeddings(inputs)
        x += self.pos_embeddings(torch.arange(x.shape[1], device=device))[None, ...]

        for transformer_module in self.transformer_modules:
            x = transformer_module(x=x, attn_mask=attn_mask)

        x = self.output_layernorm(x)
        logits = self.output_linear(x)

        return logits
