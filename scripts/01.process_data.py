## COMMAND ----------
import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from games_sales import PROJECT_DIR
from games_sales.config import ProjectConfig

# from games_sales.data_loader import DataLoader
from games_sales.data_processor import DataProcessor

if "ipykernel" in sys.modules:
    # Running interactively, mock arguments
    class Args:
        """Mock arguments used when running interactively (e.g. in Jupyter)."""

        env = "dev"
        is_test = 0

    args = Args()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--env",
        action="store",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--is_test",
        action="store",
        default=0,
        type=int,
        required=True,
    )
    args = parser.parse_args()
    PROJECT_DIR = (Path(args.root_path) / "files").resolve()


config = ProjectConfig.from_yaml(config_path=(PROJECT_DIR / "project_config.yml").resolve(), env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()
# Load data
# data_loader = DataLoader(config=config, from_volume=True, spark=spark)
# df = data_loader.load_data(config)
df = spark.read.csv("/Volumes/mlops_dev/pborys/data/vgsales.csv", header=True, inferSchema=True).toPandas()
df = df[df["Year"] != "N/A"]

logger.info("Raw data loaded")
# COMMAND ----------


# Processing
data_processor = DataProcessor(pandas_df=df, config=config, spark=spark)
data_processor.preprocess_data()

train_df, test_df = data_processor.split_by_time(
    test_periods=2, min_train_periods=int(config.preprocessing["min_years"] * 0.8)
)

# Save to catalog
data_processor.save_to_catalog(train_df, test_df)
logger.info("Data saved to catalog")
