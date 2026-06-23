import torch 
import torch.nn as nn 
from torch.nn import Embedding
import numpy as np 

class PositionEmbedding(nn.Module): 
    def __init__(self, 
                 vocab_size: int, 
                 d_model: int) -> None: 
        super().__init__()
        self._d_model: int = d_model
        self._embedding: Embedding = Embedding(vocab_size, self._d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._embedding(x) * np.sqrt(self._d_model)
    

class Positional(nn.Module): 
    def __init__(self, d_model: int, 
                 seq_len: int ): 
        super().__init__()
        d_model: int = d_model
        seq_len: int =seq_len
        pe = Positional._define_pe(d_model, seq_len)

        self.register_buffer("pe", pe)
        
    @staticmethod
    def _define_pe(d_model: int, 
                   seq_len: int):
        """ 
        Define pe argument. 
        shape: 
            (1, seq_len, d_model)
        """
        pe: torch.Tensor = torch.zeros(seq_len, d_model)
        div_term: torch.Tensor = torch.arange(0, d_model, 2, dtype= torch.float32)
        div_term = torch.exp(-np.log(1e4) * 2* div_term / d_model).unsqueeze(0) # Shape: (1, d_model)
        position: torch.Tensor = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1) # Shape: (seq_len, 1)

        position = position * div_term
        pe[:, 0::2] = torch.sin(position)
        pe[:, 1::2] = torch.cos(position)
        
        return pe.unsqueeze(0)    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return x + self.pe[:, :x.size(1), :]