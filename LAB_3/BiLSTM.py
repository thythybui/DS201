import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from typing import Tuple, Optional


class BiLSTMEncoder(nn.Module):
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, dropout: float, num_ner_tags: int):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout, 
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.output = nn.Linear(hidden_size * 2, num_ner_tags)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        init.zeros_(param.data)
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        embedded = self.embedding(x)
        
        output, (hidden, cell) = self.bilstm(embedded)
        
        x = self.dropout(output)
        
        logits = self.output(x)
        
        output = F.log_softmax(logits, dim=2)
        
        return output