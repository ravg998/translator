import torch 
import torch.nn  as nn
from .encoder import Encoder 
from .decoder import Decoder 
from .positional import Positional, PositionEmbedding 
from .output import Output 

class Transformer(nn.Module):
    def __init__(self, 
                 d_model: int,
                 n_head: int, 
                 seq_len:int, 
                 dd_df: int,
                 n_encoder_layer: int, 
                 n_decoder_layer:int,
                 vocab_size_src: int,
                 vocab_size_tgt: int): 
        super().__init__()
        self._encoder: Encoder = Encoder(n_encoder_layer=n_encoder_layer, 
                                         d_model = d_model, 
                                         n_head = n_head, 
                                         seq_len=seq_len, 
                                         dd_df=dd_df)
        self._decoder: Decoder = Decoder(n_decoder_layer=n_decoder_layer, 
                                         d_model = d_model, 
                                         n_head = n_head, 
                                         dd_df=dd_df,
                                         seq_len=seq_len) 
        self._positional_embedding_src: PositionEmbedding = PositionEmbedding(vocab_size_src,
                                                                              d_model=d_model)
        self._positional_embedding_tgt: PositionEmbedding = PositionEmbedding(vocab_size_tgt, 
                                                                              d_model=d_model)
        
        self._positional: Positional = Positional(seq_len=seq_len,
                                                  d_model=d_model
                                                  )
        self._output_layer: Output = Output(d_model = d_model,
                                            vocab_size_tgt=vocab_size_tgt)
        
        
    def forward(self, 
                x_src: torch.Tensor, 
                x_tgt: torch.Tensor, 
                mask_pad: torch.Tensor, 
                mask_hid: torch.Tensor) -> torch.Tensor: 
        x: torch.Tensor = self._positional_embedding_src(x_src)
        x = self._positional(x)
        x_tgt: torch.Tensor = self._positional_embedding_tgt(x_tgt)
        x_tgt = self._positional(x_tgt)
        
        x = self._encoder(x, mask_pad)
        output: torch.Tensor = self._decoder(q= x_tgt,
                                             kv= x, 
                                             mask_1 = mask_hid, 
                                             mask_2 = mask_pad 
                                             )
        return self._output_layer(output)
                
    
        