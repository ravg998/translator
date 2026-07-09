from tokenizers import Tokenizer
from translator.model import Transformer
from translator.config import settings 
import torch 

def filter_sentence_too_long(data: dict, 
                             max_seq_len: int,
                             language_src: str, 
                             language_tgt: str, 
                             token_src: Tokenizer, 
                             token_tgt: Tokenizer) -> dict:
    """ 
    Filters out sentences for which size is bigger than max_seq_len
    """
    new_data: dict = {"translation":[]}
    for sentences in data["translation"]: 
        if len(token_src.encode(sentences[language_src])) + 2<=max_seq_len \
            and len(token_tgt.encode(sentences[language_tgt]))+ 1<=max_seq_len:
            new_data["translation"].append({language_src: sentences[language_src], 
                                            language_tgt: sentences[language_tgt]})
            
    return new_data


def save_weight(save_name: str, 
                epoch: int,
                optimizer,
                model: Transformer,
                language_src: str, 
                language_tgt: str, 
                **hyperparameters): 
    model_states = {"hyperparameters": hyperparameters, 
                    "model": model.state_dict(), 
                    "languge_src": language_src, 
                    "language_tgt": language_tgt,
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict()
                    }    
    
    settings.data_path.weight.mkdir(parents = True, exist_ok=True)
    torch.save(model_states, 
               settings.data_path.weight / save_name)

