"""FeaturLookUp Serving module."""

import time

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput


class FeatureLookupServing:
    """Manage Feature Lookup Serving operations."""

    def __init__(self, model_name: str, endpoint_name: str, feature_table_name: str) -> None:
        """Initialize the Feature Lookup Serving Manager.

        :param model_name: Name of the model
        :param endpoint_name: Name of the endpoint
        :param feature_table_name: Name of the feature table
        """
        self.workspace = WorkspaceClient()
        self.feature_table_name = feature_table_name
        self.online_table_name = f"{self.feature_table_name}_online"
        self.model_name = model_name
        self.endpoint_name = endpoint_name

    def get_latest_model_version(self) -> str:
        """Get the latest version of the model.

        :return: Latest model version
        """
        client = mlflow.MlflowClient()
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        print(f"Latest model version: {latest_version}")
        return latest_version

    def create_or_update_online_table(self, online_store_name: str, max_retries: int = 3) -> None:
        """Create or update an online table for house features.

        :param online_store_name: Name of the online store
        :param max_retries: Maximum number of retry attempts (default: 3)
        """
        fe = FeatureEngineeringClient()
        online_store = fe.get_online_store(name=online_store_name)

        # Publish the feature table to the online store with retry logic
        for attempt in range(max_retries + 1):
            try:
                fe.publish_table(
                    online_store=online_store,
                    source_table_name=self.feature_table_name,
                    online_table_name=f"{self.feature_table_name}_online",
                )
                print(f"Successfully published table on attempt {attempt + 1}")
                return

            except Exception as e:
                if attempt < max_retries:
                    print(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in 60 seconds...")
                    time.sleep(60)  # Wait 1 minute before retry
                else:
                    print(f"All {max_retries + 1} attempts failed. Last error: {str(e)}")
                    raise e

    def deploy_or_update_serving_endpoint(
        self, version: str = "latest", workload_size: str = "Small", scale_to_zero: bool = True, wait: bool = False
    ) -> None:
        """Deploy or update the model serving endpoint in Databricks.

        :param version: Version of the model to deploy
        :param workload_size: Workload size (number of concurrent requests). Default is Small = 4 concurrent requests.
        :param scale_to_zero: If True, endpoint scales to 0 when unused
        :param wait: If True, the job will wait for the endpoint creation/update to finish
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace.serving_endpoints.list())
        entity_version = self.get_latest_model_version() if version == "latest" else version

        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version,
            )
        ]

        if not endpoint_exists:
            if wait:
                self.workspace.serving_endpoints.create_and_wait(
                    name=self.endpoint_name,
                    config=EndpointCoreConfigInput(
                        served_entities=served_entities,
                    ),
                )
            else:
                self.workspace.serving_endpoints.create(
                    name=self.endpoint_name,
                    config=EndpointCoreConfigInput(
                        served_entities=served_entities,
                    ),
                )
        else:
            if wait:
                self.workspace.serving_endpoints.update_config_and_wait(
                    name=self.endpoint_name,
                    served_entities=served_entities,
                )
            else:
                self.workspace.serving_endpoints.update_config(
                    name=self.endpoint_name,
                    served_entities=served_entities,
                )
