import torch 
from pathlib import Path
from translator.config import settings 
from translator.model import Transformer
from .device import setup_device 

def load_model(weight_file_name: str) -> Transformer: 
    weight_file: Path = settings.data_path.weight / weight_file_name
    weight = torch.load(weight_file)
    
    model = Transformer(**weight["hyperparameters"])
    model.load_state_dict(weight["model"])
    model.to(setup_device())
    
    return model 


def load_languages(weight_file_name: str) -> dict[str, str]: 
    weight_file: Path = settings.data_path.weight / weight_file_name
    weight = torch.load(weight_file)
    language_src: str= weight["languge_src"]
    language_tgt: str= weight["language_tgt"]
    
    return {"language_src": language_src, 
            "language_tgt": language_tgt}
    