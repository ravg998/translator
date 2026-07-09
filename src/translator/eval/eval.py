import torch 
from translator.config import settings
from translator.data import TokenLoad
from tokenizers import Tokenizer
from translator.utils import setup_device
from translator.model import Transformer
from pathlib import Path
from translator.utils import load_model, load_languages

def build_mask_decoder(size: int) -> torch.Tensor:
    """ 
    Build triangual lower matrice. 
    Shape: (1, size, size)
    """
    mask = torch.tril(torch.ones(size, size))
    
    return mask.unsqueeze(0)

@torch.no_grad()
def eval(weight_file: str, 
         sentence_to_translate: str
         ) -> str: 
    model = load_model(weight_file)
    languages: dict[str, str]= load_languages(weight_file)
    token_src: Tokenizer = TokenLoad(language=languages["language_src"]).get_token()
    token_tgt: Tokenizer = TokenLoad(language=languages["language_tgt"]).get_token()
    
    model.eval()
    encoder_src: torch.Tensor = torch.tensor(token_src.encode(sentence_to_translate).ids, 
                                             dtype= torch.int64)
    
    sos_token: list[int] = token_src.encode(settings.tokenizer_cfg.sos).ids
    sos_token_tgt: list[int] = token_tgt.encode(settings.tokenizer_cfg.sos).ids
    eos_token: list[int] = token_src.encode(settings.tokenizer_cfg.eos).ids
    
    encoder_src = torch.cat([torch.tensor(sos_token, dtype=torch.int64), 
                             encoder_src,
                             torch.tensor(eos_token, dtype=torch.int64)
                            ]).unsqueeze(0) # (1, LEN_SEQ)
    
    decoder = torch.tensor(sos_token_tgt, dtype=torch.int64).unsqueeze(0) # (1, 1)
    mask_input = torch.ones_like(encoder_src)
    
    eos_id_tgt = token_tgt.token_to_id(settings.tokenizer_cfg.eos)
    translation:str = ""
    while decoder[0, -1].item()!= eos_id_tgt: 
        mask_decoder = build_mask_decoder(decoder.shape[1])
        
        output = model(x_src = encoder_src.to(setup_device()), 
                       x_tgt = decoder.to(setup_device()), 
                       mask_pad = mask_input.to(setup_device()), 
                       mask_hid = mask_decoder.to(setup_device()))
        
        next_item = torch.argmax(output[0, -1, :]).item()
        decoder = torch.cat([decoder, 
                            torch.tensor([next_item], 
                                         dtype=torch.int64).unsqueeze(0)
                            ], dim = 1)
        
        translation = f"{translation} {token_tgt.decode([next_item])}"
        
    return translation 
    
    
    
    