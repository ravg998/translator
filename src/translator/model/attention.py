import torch 
import torch.nn as nn 
import numpy as np 

"""
class to Handle the Attention. It will: 
    • Apply a Linear transformation from d_model -> d_model
    • Compute Q @ K.T = QK
    • Apply the mask to QK -> QK_MASK
    • Compute Softmax(QK_MASK) = QK_SOFTMAX
    • Compute QK_SOFTMAX @ V 
    
    
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 n_head: int, 
                 d_model: int, 
                 seq_len: int) -> None: 
        super().__init__()
        
        self._n_head: int = n_head
        self._d_model: int = d_model
        self._seq_len: int = seq_len
        self._dk: int = self._d_model // self._n_head
        assert  self._d_model % self._n_head == 0, f"d_model ({self._d_model:,.0f}) cannot be divided by n_head ({self._n_head:,.0f})"
        
        # MODEL 
        self._q: nn.Linear = nn.Linear(self._d_model, self._d_model)
        self._k: nn.Linear = nn.Linear(self._d_model, self._d_model)
        self._v: nn.Linear = nn.Linear(self._d_model, self._d_model)
        self._output: nn.Linear = nn.Linear(self._d_model, self._d_model)
        

    def _reshape_input(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        Reshape in 2 steps. 
            STEP 0 INPUT SHAPE:     (batch_size, seq_len, d_model)
            STEP 1 RESHAPE SHAPE:   (batch_size, seq_len, n_head, d_k)
            STEP 2 TRANSPOSE:       (batch,size, n_head, seq_len, d_k) 

        OUTPUT SHAPE: (batch,size, n_head, seq_len, d_k)
        """
        seq_len = x.shape[1]  # longueur RÉELLE, pas la constante
        x = x.reshape(x.shape[0], 
                      seq_len,
                      self._n_head, 
                      self._dk)
        x = x.transpose(1,2)
        return x 
          
    def forward(self, 
                q: torch.Tensor, 
                k: torch.Tensor,
                v: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """ 
        Input Shape: 
            (batch_size, seq_len, d_model) 
            
        output Shape: 
            (batch_size, seq_len, d_model)
        """
        q  = self._q(q)
        k  = self._k(k)
        v = self._v(v)
        
        q = self._reshape_input(q)
        k = self._reshape_input(k)
        v = self._reshape_input(v)
        
        qk: torch.Tensor = q @ k.transpose(-1,-2) / np.sqrt(self._dk) # SHAPE: (batch_size, n_head, seq_len, seq_len)
        qk = qk.masked_fill(mask == 0, -1e9)
        
        qk = torch.softmax(qk, dim=-1)
        output: torch.Tensor = qk @ v 
        
        output = output.transpose(1,2)
        output = output.reshape(output.shape[0], 
                                -1, 
                                self._d_model)
        
        
        return self._output(output) 