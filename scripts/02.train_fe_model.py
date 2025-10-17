import argparse
from pathlib import Path

import yaml
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

# from games_sales import PROJECT_DIR
from games_sales.config import ProjectConfig, Tags
from games_sales.models.feature_lookup_model import FeatureLookUpModel

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

PROJECT_DIR = Path(args.root_path).resolve()

config = ProjectConfig.from_yaml(config_path=(PROJECT_DIR / "files/project_config.yml").resolve(), env="dev")
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)
# tags = Tags.from_git_repo(repo_path=PROJECT_DIR)
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
logger.info(yaml.dump(tags, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
logger.info("Feature LookUp Model initialized.")

# Create feature table
fe_model.create_feature_table()
logger.info("Feature table created.")

# fe_model.update_feature_table()
# logger.info("Feature table updated.")

# Load data
fe_model.load_data()
logger.info("Data loaded.")

# Perform feature engineering
fe_model.feature_engineering()

# Train the model
fe_model.train()
logger.info("Model trained.")

fe_model.log_model()

# Evaluate model
# Load test set from Delta table
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target
model_improved = fe_model.model_improved(test_set=test_set)
logger.info("Model evaluation completed, model improved: ", model_improved)

is_test = args.is_test

# when running test, always register and deploy
if is_test == args.is_test:
    model_improved = True

if model_improved:
    # Register the model
    latest_version = fe_model.register_model()
    logger.info("New model registered with version:", latest_version)
    dbutils.jobs.taskValues.set(key="model_version", value=latest_version)
    dbutils.jobs.taskValues.set(key="model_updated", value=1)
else:
    dbutils.jobs.taskValues.set(key="model_updated", value=0)
    logger.info("Model not improved, registration skipped.")
