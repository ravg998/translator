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
