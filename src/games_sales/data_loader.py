import kagglehub
import pandas as pd
from games_sales.config import ProjectConfig
from pathlib import Path
import shutil
from loguru import logger

class DataLoader:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def load_data(self, config:ProjectConfig) -> pd.DataFrame:
        """Load data into a Pandas DataFrame.
        return: DataFrame containing the loaded data
        """
        local_path = Path(config.data_source['local_path'])
        if config.data_source['force_download'] or not local_path.exists():
            download_path = kagglehub.dataset_download(config.data_source['online_path'], force_download=True)
            download_path = Path(download_path)
            logger.info(f"Dataset downloaded to {download_path}")
            files_found = list(download_path.glob(f"*{config.data_source['file_name']}"))
            if len(files_found) != 1:
                error_message = f"Either no file or multiple files in download folder {download_path} is matching {config.data_source['file_name']}"
                logger.error(error_message)
                raise AssertionError(error_message)
            else:
                shutil.move(str(files_found[0]), str(local_path))
                logger.info(f"File moved to {str(local_path)}")

        df = pd.read_csv(local_path)
        logger.info(f"Data loaded from {local_path} with shape {df.shape}")
        return df