import shutil
from pathlib import Path

import kagglehub
import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession

from games_sales.config import ProjectConfig


class DataLoader:
    """Class to handle data loading from either online kaggle source or local path."""

    def __init__(self, config: ProjectConfig, from_volume: bool, spark: SparkSession) -> None:
        self.config = config
        self.from_volume = from_volume
        self.spark = spark

    def load_data(self, config: ProjectConfig) -> pd.DataFrame:
        """Load data into a Pandas DataFrame.

        return: DataFrame containing the loaded data
        """
        if self.from_volume:
            spark = self.spark

            volume_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/data/{config.data_source['file_name']}"
            df = spark.read.csv(volume_path, header=True, inferSchema=True).toPandas()

            logger.info(f"Data loaded from volume at {volume_path} with shape {df.shape}")
            return df

        local_path = Path(config.data_source["local_path"])
        if config.data_source["force_download"] or not local_path.exists():
            download_path = kagglehub.dataset_download(config.data_source["online_path"], force_download=True)
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
