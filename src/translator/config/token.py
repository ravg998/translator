from pydantic import BaseModel, ConfigDict, Field 

    
class Token(BaseModel): 
    model_config= ConfigDict(frozen=True)
    
    sos: str = Field(description="Token for Start of Sentence", 
                     default = "[SOS]")
    eos: str = Field(description="Token for End of Sentence", 
                     default = "[EOS]")
    unk: str = Field(description="Token for Unknown words", 
                     default = "[UNK]")
    padding:str = Field(description="Token for Padding", 
                        default = "[PAD]")
    min_frequency: int = Field(description="Minium number of times for a word to appear in the dataset to be considered", 
                               ge=1, default = 1)
    @property 
    def special_token(self) -> list[str]: 
        return [self.sos, self.eos, self.unk, self.padding]
    
