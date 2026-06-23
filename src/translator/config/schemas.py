from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource, CliSettingsSource
from enum import Enum
from .config_var import PATH_WORKSPACE, PATH_CONFIG, PATH_DATA_FOLDER
from pathlib import Path
from .logger import Logger 
from .token import Token
from .datatext import DataText
from .datapath import DataPath
from .model import Model



class AppSettings(BaseSettings): 
    model_config = SettingsConfigDict(cli_parse_args=True, 
                                      yaml_file=PATH_CONFIG)
    app_name: str
    logger: Logger
    tokenizer_cfg: Token = Field(default_factory=lambda: Token())
    
    model: Model 
    data_text: DataText
    data_path: DataPath
    
    @classmethod
    def settings_customise_sources(cls, 
                                    settings_cls, 
                                    init_settings, 
                                    env_settings, 
                                    dotenv_settings, 
                                    file_secret_settings):
        return (CliSettingsSource(settings_cls, cli_parse_args=True), 
                init_settings,
                YamlConfigSettingsSource(settings_cls), 
                env_settings, 
                dotenv_settings, 
                file_secret_settings
        )
       
settings = AppSettings() 
if __name__=="__main__":
    ...
    print(settings)