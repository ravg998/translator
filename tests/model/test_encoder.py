import pytest 
import torch 
from translator.model import EncoderBlock 

parameters: list[dict[str, int]] = [
    {"seq_len": 365, 
     "n_head": 8, 
     "d_model": 64, 
     "batch_size": 8, 
     "dd_df": 2048}
]


@pytest.fixture(params= parameters)
def encoder_block(request: pytest.fixture) -> EncoderBlock: 
    param = request.param 
    return EncoderBlock(seq_len = param["seq_len"], 
                        n_head= param["n_head"], 
                        d_model= param["d_model"], 
                        dd_df=param["dd_df"])
    
@pytest.fixture(params=parameters)
def x_input(request: pytest.fixture) -> torch.Tensor:
    param = request.param
    return torch.rand((param["batch_size"], 
                       param["seq_len"], 
                       param["d_model"]))

@pytest.fixture(params=parameters)
def parameter(request: pytest.fixture) -> dict[str, int]:
    return request.param


@pytest.fixture(params=parameters)
def mask(request: pytest.fixture) -> torch.Tensor: 
    param = request.param 
    
    return torch.rand((1, 1, param["seq_len"]))

    
class TestEncoderBlock: 
    def test_forward_shape(self, 
                           encoder_block: EncoderBlock, 
                           x_input: torch.Tensor, 
                           parameter: dict[str, int], 
                           mask: torch.Tensor):
        x_output = encoder_block(x_input, 
                                 
                                 mask)
        
        assert x_output.shape ==((parameter["batch_size"], 
                                  parameter["seq_len"], 
                                  parameter["d_model"])) 
        