from tokenizers import Tokenizer
from torch.utils.data import Dataset
from translator.config import Token
import torch 

class TokenDataset(Dataset): 
    def __init__(self,
                 dataset: dict[str, list[dict[str, str]]],
                 token_src: Tokenizer, 
                 token_tgt: Tokenizer, 
                 seq_len: int, 
                 language_src: str, 
                 language_tgt: str, 
                 pad_token: str, 
                 sos_token: str, 
                 eos_token: str ):
        """ 
        dataset: dict[str, list[dict[str, str]]]: {"translation": [  
                                                    {"en": "Hello", 
                                                    "fr": "Salut"
                                                    }
                                                ]
                                }
        """
        super().__init__()
        self._dataset: dict[str, str] = dataset 
        self._token_src: Tokenizer = token_src
        self._token_tgt: Tokenizer = token_tgt
        self._language_src: str = language_src
        self._language_tgt: str = language_tgt
        self._seq_len: int  =  seq_len
        
        # TOKENS
        self._pad_token: str= pad_token
        self._sos_token: str = sos_token
        self._eos_token: str = eos_token
           
        
    def __len__(self) -> int:
        return len(self._dataset["translation"])
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ 
        Get Item will: 
            • Extracts the idx-th position from the dataset:
                - Extract language src
                - Extract language tgt
            • For both src and tgt: 
                - Encode special tokens (UNK, SOS,...)
                - Encode the text ——> SHAPE: (X)
            • For src:
                - It will add SOS and EOS tokens to encoded text ——> SHAPE: (X+2)
                - Fill with PAD tokens to have a size of seq_len ——> SHAPE: (seq_len)
                
        returns: 
            • encoder_input: (seq_len)
            • encoder_mask: (1, 1, seq_len)
            • decoder_tgt: (seq_len)
            • label: (seq_len)
            • decoder_mask: (1, seq_len, seq_len)
        """
        text_src: str = self._dataset["translation"][idx][self._language_src]
        text_tgt: str = self._dataset["translation"][idx][self._language_tgt]
        
        # SPECIAL TOKENS 
        sos_token_src, eos_token_src, pad_token_src = self._tokenize_special_token(self._token_src)
        sos_token_tgt, eos_token_tgt, pad_token_tgt = self._tokenize_special_token(self._token_tgt)
   
        # ENCODER 
        encoder_src = self._define_encoder_input(text_src, sos_token_src, eos_token_src, pad_token_src) # SHAPE: (seq_len)
        
        # DECODER
        decoder_tgt, label = self._define_decoder_input(text_tgt, 
                                                        sos_token_tgt, eos_token_tgt, pad_token_tgt)
        
        assert len(encoder_src) == self._seq_len
        assert len(decoder_tgt) == self._seq_len
        assert len(label) == self._seq_len
        
        
        # MASK 
        encoder_mask: torch.Tensor = self._define_mask_padding(encoder_src, self._token_src).unsqueeze(0) # (1, 1, seq_len)
        decoder_mask: torch.Tensor =  self._define_decoder_mask(decoder_tgt).unsqueeze(0)
        
        
        return {"encoder_input": encoder_src, 
                "encoder_mask": encoder_mask,
                "decoder_tgt": decoder_tgt, 
                "label": label, 
                "decoder_mask": decoder_mask 
                }
    
    def _tokenize_special_token(self, token: Tokenizer) -> list[torch.Tensor]: 
        """ 
        Tokenize special tokens and turn them into Tensor. 
        """
        sos_token: torch.Tensor =  torch.tensor(token.encode(self._sos_token).ids, dtype = torch.int32)
        eos_token: torch.Tensor =  torch.tensor(token.encode(self._eos_token).ids, dtype = torch.int32)
        pad_token: torch.Tensor =  torch.tensor(token.encode(self._pad_token).ids, dtype = torch.int32)
        
        return sos_token, eos_token, pad_token
        
    def _define_encoder_input(self, 
                              text_src: str, 
                              sos_token: torch.Tensor, eos_token: torch.Tensor, pad_token: torch.Tensor):
        """ 
        Encodes text source
        Add SOS and EOS to token 
        Fill gap with PAD token to reach seq_len size. 
        
        Raises an error if sentence is too big.
        
        returns: torch.Tensor of shape: (SEQ_LEN)
        """
        encoder_src: torch.Tensor = torch.tensor(self._token_src.encode(text_src).ids, 
                                                 dtype = torch.int32)
        encoder_src = torch.cat([sos_token, encoder_src, eos_token], 
                                dim=0)
        size_pad_token: int = self._seq_len - len(encoder_src)
        assert size_pad_token >=0, f"Sentence too long: Max size allowed is {self._seq_len:,.0f} while input sentence is {len(encoder_src):,.0f}"
        
        encoder_src = torch.cat([encoder_src] + [pad_token] * size_pad_token, dim=0)
        
        return encoder_src 
        
    def _define_decoder_input(self, text_tgt: str, sos_token: torch.Tensor, eos_token: torch.Tensor, pad_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ 
        Encodes text target. 
        Returns 2 tensors: 
            • decoder_tgt: tensor that will be used as an input for the decoder section. 
            • label: tensor that will be used for results. 
            
        Shape of each tensor: (seq_len)
        """
        all_tgt_ids: list[int] = self._token_tgt.encode(text_tgt).ids 
        all_tgt_tensor: torch.Tensor = torch.tensor(all_tgt_ids, dtype = torch.int32) # SIZE: (X)
        
        # DECODER INPUT
        decoder_tgt: torch.Tensor = torch.cat([sos_token, all_tgt_tensor], dim=0)
        
        pad_decoder_len: int = self._seq_len - len(decoder_tgt)
        decoder_tgt = torch.cat([decoder_tgt] + [pad_token] * pad_decoder_len, dim = 0) # SHAPE: (seq_len)
        
        # LABEL 
        label: torch.Tensor = torch.cat([all_tgt_tensor, eos_token], dim=0)
        pad_decoder_len: int = self._seq_len - len(label)
        label = torch.cat([label] + [pad_token] * pad_decoder_len, dim = 0) # SHAPE: (seq_len)
        
        return decoder_tgt, label
        
        
        
        
        
    def _define_mask_padding(self, sequence: torch.Tensor, tokenizer: Tokenizer) -> torch.Tensor:
        """ 
        Crée un masque de padding pour une séquence donnée.
        Shape: 
            (1, seq_len)
        """
        padding_token: int = torch.tensor(tokenizer.encode(self._pad_token).ids)
        mask =  sequence != padding_token
        return mask.unsqueeze(0)
        
    def _define_decoder_mask(self, decoder: torch.Tensor) -> torch.Tensor:
        """ 
        Combine masque de padding (cible) + masque causal.
        Shape: 
            (seq_len, seq_len)
        """
        mask_hide_future = torch.tril(torch.ones(self._seq_len, self._seq_len)) # (seq_len, seq_len)
        mask = self._define_mask_padding(decoder, self._token_tgt)
        
        return mask & (mask_hide_future==1)
        
        
if __name__=="__main__": 
    print(torch.tril(torch.ones((4,4)), diagonal=-1
                     )
          )
        
        
        
        
        