"""Basic model implementation.

infer_signature (from mlflow.models) → Captures input-output schema for model tracking.

num_features → List of numerical feature names.
cat_features → List of categorical feature names.
target → The column to predict.
parameters → Hyperparameters for LightGBM.
catalog_name, schema_name → Database schema names for Databricks tables.
"""

import os

import mlflow
import numpy as np
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup  # FeatureFunction
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from games_sales.config import ProjectConfig, Tags
from games_sales.utils import is_databricks


class FeatureLookUpModel:
    """A model with feature lookup for games sales prediction using LightGBM.

    This class handles data loading, feature preparation, model training, and MLflow logging.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, log_on_dbx: bool = True) -> None:
        """Initialize the model with project configuration.

        :param config: Project configuration object
        :param tags: Tags object
        :param spark: SparkSession object
        :param log_on_dbx: Whether to log the model on Databricks - relevant for local runs
        """
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.features_from_lookup = self.config.features_from_lookup
        self.target = self.config.target_column
        self.parameters = self.config.model_parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_fe
        self.model_name = f"{self.catalog_name}.{self.schema_name}.games_sales_model_fe"

        self.tags = tags.dict()

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.games_sales_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_age"

        if not is_databricks():
            raise OSError("FeatureLookUpModel can only be run in a Databricks environment.")

        if not is_databricks() and log_on_dbx:
            load_dotenv()
            profile = os.environ.get("PROFILE_NAME")
            mlflow.set_tracking_uri(f"databricks://{profile}")
            mlflow.set_registry_uri(f"databricks-uc://{profile}")
            logger.info(f"MLflow tracking URI set to Databricks with profile {profile}")

    def create_feature_table(self) -> None:
        """Create or update the house_features table and populate it.

        This table stores features related to houses.
        """
        # Create column definitions dynamically
        feature_columns = ", ".join(f"{feature} DOUBLE" for feature in self.features_from_lookup)
        create_table_sql = f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (Id STRING NOT NULL, {feature_columns});
        """
        self.spark.sql(create_table_sql)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT games_pk PRIMARY KEY(Id);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        # Create column list for SELECT statement
        select_columns = ", ".join(["Id"] + self.features_from_lookup)
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT {select_columns} FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT {select_columns} FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading data from Databricks tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas().drop(columns=self.features_from_lookup)
        self.train_set["Id"] = self.train_set["Id"].astype(str)

        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set")
        self.test_set = self.test_set.toPandas()
        self.test_set["Id"] = self.test_set["Id"].astype(str)

        cols_X = [col for col in self.num_features + self.cat_features if col not in self.features_from_lookup]
        self.X_train = self.train_set[cols_X]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[cols_X]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        # Check if train_set is a Spark DataFrame or Pandas
        if isinstance(self.train_set, pd.DataFrame):
            train_set_spark = self.spark.createDataFrame(self.train_set)
        else:
            train_set_spark = self.train_set

        self.training_set = self.fe.create_training_set(
            df=train_set_spark,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=self.features_from_lookup,
                    lookup_key="Id",
                ),
                # FeatureFunction(
                #     udf_name=self.function_name,
                #     output_name="house_age",
                #     input_bindings={"year_built": "YearBuilt"},
                # ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        # current_year = datetime.now().year
        # self.test_set["house_age"] = current_year - self.test_set["YearBuilt"]

        self.X_train = self.training_df[self.num_features + self.cat_features]  # + ["house_age"]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]  # + ["house_age"]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Encode categorical features and train a model pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM regression model.
        """
        logger.info("🔄 Defining model pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("regressor", LGBMRegressor(**self.parameters))]
        )
        logger.info("✅ Preprocessing pipeline defined.")

        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("✅ Model pipeline training complete.")

    def log_model(self) -> None:
        """Log the model using MLflow."""
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            # Evaluate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            # Log the model
            signature = infer_signature(model_input=self.X_train, model_output=y_pred)

            self.fe.log_model(
                model=self.pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("🔄 Registering the model in UC...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=self.model_name,
            tags=self.tags,
        )
        logger.info(f"✅ Model registered as version {registered_model.version}.")

        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=self.model_name,
            alias="latest-model",
            version=latest_version,
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        logger.info("✅ Dataset source loaded.")
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        :return: Tuple containing metrics and parameters dictionaries
        """
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        logger.info("✅ Dataset metadata loaded.")
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading model from MLflow alias 'production'...")

        model_uri = f"models:/{self.model_name}@latest-model"
        predictions = self.fe.score_batch(model_uri=model_uri, df=input_data)

        logger.info("✅ Predictions generated.")
        # Return predictions as a DataFrame
        return predictions
