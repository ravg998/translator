from datasets import load_dataset, load_from_disk, Dataset
from pathlib import Path 
import logging

logger = logging.getLogger(__name__)

""" 
Extract data set from dataset package. 
Output is a Dataset with following iterator: 
    * id
    * translation:  language_src
                    language_tgt
"""

def download_dataset(language_src: str, 
                     language_tgt: str, 
                     source: str) -> Dataset:
    return load_dataset(source, f"{language_src}-{language_tgt}", 
                        split = "train")


def open_data(data_path: Path) -> Dataset:
    return load_from_disk(data_path)
        
class DataSource:
    def __init__(self, 
                 language_src: str, 
                 language_tgt:  str, 
                 data_source: str, 
                 save_path: Path): 
        self._language_src: str = language_src
        self._language_tgt: str = language_tgt
        self._data_source: Path = data_source
        self._save_path: Path = save_path / f"{self._language_src}_to_{self._language_tgt}"
        
    def get_data(self) -> dict:
        """ 
        Output is a Dataset with following iterator: 
            * id
            * translation:  language_src
                            language_tgt
            
        """
        if self._save_path.exists():
            logger.info(f"Extract data from local memory")
            return open_data(self._save_path)
        
        return self._download_data()
    
    def _download_data(self) -> dict:
        logger.info(f"Download dataset {self._data_source}: from {self._language_src} to {self._language_tgt}...")
        data = download_dataset(self._language_src, 
                                self._language_tgt, 
                                self._data_source)
        
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        data.save_to_disk(self._save_path)
        
        logger.info(f"... dataset downloaded succeessfully. Data saved to {self._save_path} file.")
        
        return data
    
if __name__=="__main__":
    from translator.utils import setup_logger 
    setup_logger()
    data_source = DataSource()
    data = data_source.get_data()
    print(data["translation"][0])
  