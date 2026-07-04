from tokenizers import Tokenizer
from tokenizers.models import WordLevel 
from tokenizers.pre_tokenizers import Whitespace 
from tokenizers.trainers import WordLevelTrainer 
import logging
from translator.config import settings, Token
from pathlib import Path


logger = logging.getLogger(__name__)

def iterate_over_data(data, 
                      language: str):
    for d in data["translation"]: 

        yield d[language]
        
         
class TokenLoad: 
    def __init__(self, 
                 language: str, 
                 data: dict | None = None): 
        self._data: dict | None = data 
        self._language: str = language
        self._save_path: Path = settings.data_path.token / f"token_{self._language}"
        self._token_cfg: Token = settings.tokenizer_cfg
        
    def get_token(self, force_load: bool = False) -> Tokenizer:
        """ 
        Output:
            Token
        """
        if self._save_path.exists() and not(force_load):
            token =  Tokenizer.from_file(str(self._save_path))
            
        else:
            if self._data is None: 
                raise AttributeError("data is None. Need it to create the token.")
            
            token =  self._generate_token()
            self._save_path.parent.mkdir(parents = True, exist_ok=  True)
            token.save(str(self._save_path))
            
        return token 
        
    
    def _generate_token(self) -> Tokenizer:
        token = Tokenizer(WordLevel(unk_token = self._token_cfg.unk))
        token.pre_tokenizer = Whitespace() # SPLIT TOKEN USING WHITE SPACE 
        trainer = WordLevelTrainer(special_tokens=self._token_cfg.special_token, 
                                   min_frequency= self._token_cfg.min_frequency)
        
        token.train_from_iterator(iterate_over_data(self._data, 
                                                    self._language), 
                                  trainer = trainer) 
        
        return token
    
if __name__=="__main__": 
    from .extract_data import DataSource 
    import torch 
    data = DataSource().get_data()
    
    token_load= TokenLoad(data, 
                          settings.data_text.language_src)
    
    token = token_load.get_token()
    word_to_encode: str =data["translation"][895]["en"]
    print(word_to_encode)
    
    tensor  = torch.tensor(
                        token.encode(word_to_encode).ids, 
                        dtype = torch.int32
                       )
          
    print(torch.cat([tensor, torch.tensor([1200000, -909])]) == 
          torch.cat([tensor, torch.tensor([1200000, -909])], dim=0)
          )