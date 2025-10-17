# Databricks notebook source
# MAGIC %pip install games_sales-1.0.0-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import os
import time
from pathlib import Path

import requests
import yaml
from databricks.connect import DatabricksSession
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from loguru import logger

from games_sales import PROJECT_DIR
from games_sales.config import ProjectConfig, Tags
from games_sales.models.feature_lookup_model import FeatureLookUpModel
from games_sales.serving.fe_model_serving import FeatureLookupServing
from games_sales.utils import is_databricks

if is_databricks():
    PROJECT_DIR = Path.cwd().parent.resolve()

# COMMAND ----------


spark = DatabricksSession.builder.profile(os.environ.get("PROFILE_NAME")).getOrCreate()

w = WorkspaceClient()
# w = WorkspaceClient(profile=os.environ.get("PROFILE_NAME"))
# os.environ["DBR_HOST"] = w.config.host  # TODO Working only on DBX Serverless :(
os.environ['DBR_HOST'] = spark.conf.get("spark.databricks.workspaceUrl")
os.environ["DBR_TOKEN"] = w.tokens.create(lifetime_seconds=1200).token_value  # TODO Not working from VS Code


# Load project config
endpoint_name = "games-sales-model-serving-fe"
config = ProjectConfig.from_yaml(config_path=(PROJECT_DIR / "project_config.yml").resolve(), env="dev")
tags = Tags.from_git_repo(repo_path=PROJECT_DIR)
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
logger.info(yaml.dump(tags, default_flow_style=False))


# COMMAND ----------
# Initialize Feature Lookup Serving Manager

model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

feature_model_server = FeatureLookupServing(
    model_name=model.model_name, endpoint_name=endpoint_name, feature_table_name=model.feature_table_name
)

# COMMAND ----------
# Create online store
fe = FeatureEngineeringClient()
online_store_name = "games-sales-predictions"
if fe.get_online_store(name=online_store_name) is None:
    fe.create_online_store(name=online_store_name, capacity="CU_1")
    online_store = fe.get_online_store(name=online_store_name)
else:
    online_store = fe.get_online_store(name=online_store_name)
# COMMAND ----------

# Create the online table for games sales features
feature_model_server.create_or_update_online_table(online_store_name=online_store_name)

# COMMAND ----------
# Deploy the model serving endpoint with feature lookup
feature_model_server.deploy_or_update_serving_endpoint()


# COMMAND ----------
# Create a sample request body

required_columns = [
    col for col in config.num_features + config.cat_features if col not in config.features_from_lookup
] + ["Id"]

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
sampled_records = train_set[required_columns].sample(n=1000, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

logger.info(train_set.dtypes)
logger.info(dataframe_records[0])


# COMMAND ----------
# Call the endpoint with one sample record
def call_endpoint(record: list[dict]) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/{endpoint_name}/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------
# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
