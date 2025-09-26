import os

import yaml
from databricks.connect import DatabricksSession
from loguru import logger

from games_sales import PROJECT_DIR
from games_sales.config import ProjectConfig, Tags
from games_sales.models.basic_model import BasicModel

config = ProjectConfig.from_yaml(config_path=(PROJECT_DIR / "project_config.yml").resolve(), env="dev")
tags = Tags.from_git_repo(repo_path=PROJECT_DIR)
logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))
logger.info(yaml.dump(tags, default_flow_style=False))

spark = DatabricksSession.builder.profile(os.environ.get("PROFILE_NAME")).getOrCreate()

# Initialize the basic model
model = BasicModel(config=config, tags=tags, spark=spark, log_on_dbx=True)

# Load data
model.load_data()

# Prepare features
model.prepare_features()

# Train and log the model
model.train()

model.log_model()

model.register_model()
