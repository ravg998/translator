from pathlib import Path 
from pydantic import BaseModel, ConfigDict, field_validator 
from .config_var import PATH_DATA_FOLDER

 
class DataPath(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    weight: Path
    data_source: Path 
    output: Path
    token: Path
    
    @field_validator("weight","data_source","output","token", mode="before" )
    @classmethod
    def _assign_full_data_path(cls, main_path: str) -> Path:
        return PATH_DATA_FOLDER / main_path 