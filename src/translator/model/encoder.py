import torch 
import torch.nn as nn 
from .attention import MultiHeadAttention 
from .feedforward import FeedForward

""" 
Encoder block: 
    • Compute Multi Head Self Attention
    • Normalize results
"""
class EncoderBlock(nn.Module): 
    def __init__(self, 
                 n_head: int, 
                 seq_len: int,
                 d_model: int, 
                 dd_df: int): 
        super().__init__()
        self._multi_head_attention: MultiHeadAttention = MultiHeadAttention(n_head= n_head, 
                                                                            d_model=d_model, 
                                                                            seq_len = seq_len) 
        self._feed_forward: FeedForward = FeedForward(d_model = d_model, 
                                                      dd_df= dd_df)
        self._norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self._norm2: nn.LayerNorm = nn.LayerNorm(d_model)

        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self._norm1(x + self._multi_head_attention(q=x, k=x, v=x, mask=mask))
        x = self._norm2(x + self._feed_forward(x))
        return x
        
        
class Encoder(nn.Module): 
    def __init__(self, 
                 d_model: int,
                 dd_df:int,
                 n_head: int, 
                 seq_len: int,
                 n_encoder_layer: int): 
        super().__init__()
        self._encoder: nn.ModuleList = nn.ModuleList([EncoderBlock(d_model = d_model, 
                                                                   n_head= n_head, 
                                                                   seq_len = seq_len, 
                                                                   dd_df= dd_df) 
                                                      for _ in range(n_encoder_layer)
                                                      ]
                                                     )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: 
        for layer in self._encoder: 
            x = layer(x, mask)
            
        return x
    