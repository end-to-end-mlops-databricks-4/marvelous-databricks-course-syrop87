import argparse
from pathlib import Path

import yaml
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from games_sales.config import ProjectConfig
from games_sales.serving.fe_model_serving import FeatureLookupServing

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
logger.info(yaml.dump(config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
model_version = dbutils.jobs.taskValues.get(taskKey="train_model", key="model_version")

catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = f"games-sales-model-serving-fe-{args.env}"

# Initialize Feature Lookup Serving Manager

feature_model_server = FeatureLookupServing(
    model_name=f"{catalog_name}.{schema_name}.games_sales_model_fe",
    endpoint_name=endpoint_name,
    feature_table_name=f"{catalog_name}.{schema_name}.games_sales_features",
)

# Create the online table for games sales features
online_store_name = "games-sales-predictions"
feature_model_server.create_or_update_online_table(online_store_name=online_store_name)

# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()

logger.info("Started deployment/update of the serving endpoint.")

# Delete endpoint if test
if args.is_test == 1:
    workspace = WorkspaceClient()
    workspace.serving_endpoints.delete(name=endpoint_name)
    logger.info("Deleting serving endpoint.")
