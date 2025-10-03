# Databricks notebook source
# MAGIC %pip install games_sales-0.0.2-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python
# COMMAND ----------

from pathlib import Path

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from games_sales import PROJECT_DIR
from games_sales.config import ProjectConfig, Tags
from games_sales.models.feature_lookup_model import FeatureLookUpModel
from games_sales.utils import is_databricks

if is_databricks():
    PROJECT_DIR = Path.cwd().parent.resolve()

config = ProjectConfig.from_yaml(config_path=(PROJECT_DIR / "project_config.yml").resolve(), env="dev")
tags = Tags.from_git_repo(repo_path=PROJECT_DIR)
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
logger.info(yaml.dump(tags, default_flow_style=False))

# spark = DatabricksSession.builder.profile(os.environ.get("PROFILE_NAME")).getOrCreate()
spark = SparkSession.builder.getOrCreate()  # FE model can be run only on DBX
# Initialize the feature lookup model
# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define house age feature function
# fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()

fe_model.log_model()


# COMMAND ----------

# Train the model
fe_model.register_model()

# COMMAND ----------

# Lets run prediction on the last production model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target
X_test = test_set.drop(*config.features_from_lookup, config.target_column)


# COMMAND ----------

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)

# COMMAND ----------
