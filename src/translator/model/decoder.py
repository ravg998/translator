import torch 
import torch.nn as nn 
from .attention import MultiHeadAttention
from .feedforward import FeedForward

""" 
Decoder block:
    • Compute Multi Head attention between tgt and itself
    • Normalize 
    • Compute Multi Head MASKED attention betwwen tgt and results from encoder 
    • Normalize
"""
class DecoderBlock(nn.Module): 
    def __init__(self, 
                 seq_len: int, 
                 n_head: int,
                 d_model: int,
                    dd_df: int): 
        super().__init__()
        self._multi_head_masked: MultiHeadAttention = MultiHeadAttention(n_head=n_head, 
                                                                         seq_len= seq_len, 
                                                                         d_model=d_model)
        self._multi_head: MultiHeadAttention = MultiHeadAttention(n_head=n_head, 
                                                                         seq_len= seq_len, 
                                                                         d_model=d_model)
        self._feed_forward: FeedForward = FeedForward(d_model = d_model, 
                                                      dd_df= dd_df)
        self._norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self._norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self._norm3: nn.LayerNorm = nn.LayerNorm(d_model)
        
    def forward(self, 
                q: torch.Tensor,
                kv: torch.Tensor, 
                mask_1: torch.Tensor, 
                mask_2: torch.Tensor) -> torch.Tensor: 
        q = self._norm1(q + self._multi_head_masked(q=q, k=q, v=q, mask=mask_1))
        q = self._norm2(q + self._multi_head(q=q, k=kv, v=kv, mask=mask_2))
        q = self._norm3(q + self._feed_forward(q))
        return q
    
    
class Decoder(nn.Module): 
    def __init__(self, 
                 seq_len: int, 
                 n_head: int,
                 d_model: int,
                 dd_df: int,
                 n_decoder_layer: int): 
        super().__init__()
        self._decoders: nn.ModuleList= nn.ModuleList([DecoderBlock(n_head=n_head, 
                                                                         seq_len= seq_len, 
                                                                         d_model=d_model, 
                                                                         dd_df=dd_df) 
                                                      for _ in range(n_decoder_layer)
                                                      ])
        
    def forward(self, q: torch.Tensor, 
                kv: torch.Tensor, 
                mask_1: torch.Tensor, 
                mask_2: torch.Tensor) -> torch.Tensor:
        for layer in self._decoders: 
            q = layer(q=q, 
                      kv=kv,
                      mask_1=mask_1, 
                      mask_2 = mask_2)
            
        return q