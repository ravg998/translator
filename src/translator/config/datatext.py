from pydantic import BaseModel, ConfigDict, Field 

class DataText(BaseModel): 
    model_config = ConfigDict(frozen=True)
    
    language_src: str = Field(description="Source Language")
    language_tgt: str = Field(description = "Target Language")
    data_source: str = Field(description="Link of Hugghing Face data source", 
                             examples=["Helsinki-NLP/opus_books"])