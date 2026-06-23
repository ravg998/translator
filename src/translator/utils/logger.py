import logging 
from translator.config import settings, logger, PATH_WORKSPACE
import sys 
from pathlib import Path 

def setup_logger():
    logger_settings: logger = settings.logger
    log_path: Path = PATH_WORKSPACE / "log"
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logger_settings.log_level,
        format=logger_settings.log_format,
        datefmt=logger_settings.datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),       # console
            logging.FileHandler(log_path / f"{settings.app_name}.log"),     # fichier
        ],
    )
    