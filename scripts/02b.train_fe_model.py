# COMMAND ----------
# Databricks notebook source
# MAGIC %pip install house_price-0.0.1-py3-none-any.whl

# COMMAND ----------

import yaml
from loguru import logger
from pyspark.sql import SparkSession

from games_sales import PROJECT_DIR
from games_sales.config import ProjectConfig, Tags
from games_sales.models.feature_lookup_model import FeatureLookUpModel

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

# Register the model
fe_model.register_model()
