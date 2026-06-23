import pytest 
from translator.model import MultiHeadAttention 
import torch 

parameters: list[dict[str, int]] = [{"n_head": 8, 
                                     "d_model": 64, 
                                     "seq_len": 352, 
                                     "batch_size": 3}] # N_HEAD, D_MODEL, SEQ_LEN


@pytest.fixture(params = parameters)
def multi_head_attention(request: pytest.fixture) -> MultiHeadAttention: 
    param = request.param
    
    return MultiHeadAttention(n_head=param["n_head"], 
                              d_model=param["d_model"], 
                              seq_len=param["seq_len"]
                              )
    
@pytest.fixture(params = parameters)
def x_input(request: pytest.fixture) -> torch.Tensor:  
    param = request.param
    
    return torch.rand((param["batch_size"], 
                       param["seq_len"], 
                       param["d_model"]))

@pytest.fixture(params= parameters)
def mask(request: pytest.fixture) -> torch.Tensor:
    param = request.param
    return torch.rand((1, param["seq_len"]))


@pytest.fixture(params = parameters)
def parameter(request: pytest.fixture) -> dict[str, int]:
    return request.param 

class TestAttention:
    def test_reshape_output_shape(self, x_input: torch.Tensor, 
                            multi_head_attention: MultiHeadAttention, 
                            parameter): 
        x_transpose = multi_head_attention._reshape_input(x_input)
        
        assert x_transpose.shape == (parameter["batch_size"], 
                                     parameter["n_head"], 
                                     parameter["seq_len"], 
                                     parameter["d_model"]//parameter["n_head"])
        
    def test_forward_shape(self, x_input: torch.Tensor, 
                                multi_head_attention: MultiHeadAttention, 
                                mask: torch.Tensor,
                                parameter): 
            output = multi_head_attention(x_input, 
                                          x_input, 
                                          x_input, 
                                          mask
                                          )
            
            assert output.shape == (parameter["batch_size"], 
                                        parameter["seq_len"], 
                                        parameter["d_model"])