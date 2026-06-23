from pydantic import BaseModel, ConfigDict, Field

class Logger(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    log_level: str = Field(description="Level of the Log", 
                           default = "INFO")
    log_format: str = Field(description="Format of the Log", 
                            default=  "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    datefmt: str = Field(description="Format of the Log date", 
                         default = "%Y-%m-%d %H:%M:%S")