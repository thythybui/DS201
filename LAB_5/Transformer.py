import torch
import torch.nn as nn
import math

def  generate_padding_mask(input_ids: torch.Tensor, pad_idx: int):
    """
    input_ids: (B, T)
    return:    (B, 1, 1, T)
    """
    mask = (input_ids != pad_idx)
    return mask.unsqueeze(1).unsqueeze(2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        query, key, value: (B, T, d_model)
        attention_mask:    (B, 1, 1, T)
        """

        B, T, _ = query.shape

        # (B, T, d_model) → (B, num_heads, T, head_dim)
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention score
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum
        context = torch.matmul(attn_weights, v)

        # (B, num_heads, T, head_dim) → (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.out_proj(context)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("positional_encoding", pe)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        """
        x = x + self.positional_encoding[:, :x.size(1)]
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        # Self-attention
        attn_out = self.self_attention(x, x, x, attention_mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))

        return x
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor):
        """
        x: (B, T, d_model)
        """
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x
