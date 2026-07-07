import torch 
import torch.nn  as nn 


class FeedForward(nn.Module): 
    def __init__(self, 
                 d_model: int, 
                 dd_df: int, 
                 dropout: float): 
        super().__init__()
        d_model: int =  d_model
        dd_df: int = dd_df
        
        # SEQUENTIAL MODEL
        self._ff_model: nn.Sequential = nn.Sequential(nn.Linear(d_model, dd_df), 
                                                      nn.ReLU(), 
                                                      nn.Linear(dd_df, d_model))
        self._dropout: nn.Dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self._dropout(self._ff_model(x))
    
