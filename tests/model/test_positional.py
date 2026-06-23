import torch 
from translator.model import PositionEmbedding 
import pytest

parameters: list[dict[str, int]] = [{"batch_size": 8,
                                     "seq_len": 352,
                                     "d_model": 64, 
                                     "vocab_size": 10_000
                                    }
                                    ]


@pytest.fixture(params=parameters)
def positional_embedding(request: pytest.fixture) -> PositionEmbedding: 
    param= request.param 
    
    return PositionEmbedding(d_model=param["d_model"], 
                  vocab_size=param["vocab_size"])
    
    
@pytest.fixture(params=parameters)
def parameter(request: pytest.fixture) -> dict[str, int]: 
    return request.param 


@pytest.fixture(params=parameters)
def x_input(request: pytest.fixture) -> torch.Tensor: 
    param = request.param 
    
    return torch.arange(1, param["batch_size"] *  
                       param["seq_len"]+1).view(param["batch_size"], -1)
                      
    
    
class TestPositionEmbedding: 
    def test_position_embedding_shape(self, 
                          positional_embedding:PositionEmbedding, 
                          parameter: dict[str, int], 
                          x_input: torch.Tensor): 
        x_output = positional_embedding(x_input)
        
        assert x_output.shape == (parameter["batch_size"], 
                                  parameter["seq_len"], 
                                  parameter["d_model"])