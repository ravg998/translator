import torch 
from translator.model import Output 
import pytest

parameters: list[dict[str, int]] = [{"batch_size": 8,
                                     "seq_len": 352,
                                     "d_model": 64, 
                                     "vocab_size_tgt": 10_000
                                    }
                                    ]


@pytest.fixture(params=parameters)
def output(request: pytest.fixture) -> Output: 
    param= request.param 
    
    return Output(d_model=param["d_model"], 
                  vocab_size_tgt=param["vocab_size_tgt"])
    
    
@pytest.fixture(params=parameters)
def parameter(request: pytest.fixture) -> dict[str, int]: 
    return request.param 


@pytest.fixture(params=parameters)
def x_input(request: pytest.fixture) -> torch.Tensor: 
    param = request.param 
    
    return torch.rand((param["batch_size"], 
                       param["seq_len"], 
                       param["d_model"]))
    
    
class TestOutput: 
    def test_output_shape(self, 
                          output:Output, 
                          parameter: dict[str, int], 
                          x_input: torch.Tensor): 
        x_output = output(x_input)
        
        assert x_output.shape == (parameter["batch_size"], 
                                  parameter["seq_len"], 
                                  parameter["vocab_size_tgt"])