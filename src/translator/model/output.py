import torch 
import torch.nn as nn 

class Output(nn.Module): 
    def __init__(self, 
                 d_model: int,
                 vocab_size_tgt: int): 
        super().__init__()
        d_model: int =  d_model
        self._linear: nn.Linear = nn.Linear(d_model, vocab_size_tgt)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor: 
        return self._linear(x)
    
