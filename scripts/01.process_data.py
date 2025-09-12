import yaml
from loguru import logger
from pyspark.sql import SparkSession

from games_sales import PROJECT_DIR
from games_sales.config import ProjectConfig
from games_sales.data_loader import DataLoader
from games_sales.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path=(PROJECT_DIR / "project_config.yml").resolve(), env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

# Load data
data_loader = DataLoader(config)
df = data_loader.load_data(config)
logger.info("Raw data loaded")


# Processing
data_processor = DataProcessor(pandas_df=df, config=config, spark=spark)
data_processor.preprocess_data()

train_df, test_df = data_processor.split_by_time(
    test_periods=2, 
    min_train_periods=int(config.preprocessing['min_years']*0.8))

logger.info("Training set shape: %s", train_df.shape)
logger.info("Test set shape: %s", test_df.shape)

# Save to catalog
data_processor.save_to_catalog(train_df, test_df)
logger.info("Data saved to catalog")

