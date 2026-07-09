from pydantic import BaseModel, ConfigDict, Field, model_validator 


class Training(BaseModel): 
    lr: float = Field(description="Learning Rate - must be positive", 
                      gt=0)
    epochs: int = Field(description="Number of Epochs", 
                        ge = 1)
    batch_size: int = Field(description="Size of the Batch", 
                            ge=1)
    
    dropout: float = Field(description="Dropout Value for Training.", 
                           ge=0, le=1)
    load_prev_weight: bool = Field(description="Boolean if wants to continue training from previous run. ")
    

class Model(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    training: Training 
    
    d_model: int = Field(description="Size of the Model", 
                         gt=0)
    dd_df: int = Field(description="Size of the FeedForward Model", 
                         gt=0)
    
    
    seq_len: int =  Field(description="Max size of each Sequence length",
                          ge=1)
    n_encoder_layer: int = Field(description="Number of Encoder Blocks", 
                                 ge=1)
    
    n_decoder_layer: int = Field(description="Number of Decoder Blocks", 
                                 ge=1)
    
    n_head: int = Field(description = "Number of Head for the multi head attention", 
                        ge=1)
    
    save_model_name: str = Field(description="Name of the weigh model to be saved.")
    
    @model_validator(mode="after")
    def _check_d_model_mult_n_head(self) -> None: 
        assert self.d_model % self.n_head == 0, f"{self.d_model:,.0f} is not divided by {self.n_head:,.0f}."

        return self 
    