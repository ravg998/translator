from pathlib import Path 

PATH_WORKSPACE: Path = Path(__file__).parent.parent.parent.parent.resolve()


""" 
    FOLDER 
"""
PATH_CONFIG_FOLDER: Path= PATH_WORKSPACE / "config"
PATH_DATA_FOLDER: Path= PATH_WORKSPACE / "data"

""" 
    CONFIG FILE 
"""
PATH_CONFIG: Path= PATH_CONFIG_FOLDER / "config.yaml"

if __name__=="__main__":
    print(PATH_CONFIG)
    print(PATH_CONFIG.exists())