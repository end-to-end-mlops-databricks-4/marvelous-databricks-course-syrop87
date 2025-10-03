"""Unit tests for DataProcessor."""

import pandas as pd
import pytest
from pyspark.sql import SparkSession

from games_sales.config import ProjectConfig
from games_sales.data_processor import DataProcessor
from tests.conftest import CATALOG_DIR


def test_data_ingestion(sample_data: pd.DataFrame) -> None:
    """Test the data ingestion process by checking the shape of the sample data.

    Asserts that the sample data has at least one row and one column.

    :param sample_data: The sample data to be tested
    """
    assert sample_data.shape[0] > 0
    assert sample_data.shape[1] > 0


def test_dataprocessor_init(
    sample_data: pd.DataFrame,
    config: ProjectConfig,
    spark_session: SparkSession,
) -> None:
    """Test the initialization of DataProcessor.

    :param sample_data: Sample DataFrame for testing
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    assert isinstance(processor.df_raw, pd.DataFrame)
    assert processor.df_raw.equals(sample_data)

    assert isinstance(processor.config, ProjectConfig)
    assert isinstance(processor.spark, SparkSession)


def test_preprocess_data(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the preprocess_data method of DataProcessor.

    This function verifies that the preprocess_data method correctly processes
    the input DataFrame according to the configuration.

    :param sample_data: Input DataFrame containing sample data
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)

    # Call the preprocess_data method
    processor.preprocess_data()

    # Verify that the DataFrame has been processed
    assert isinstance(processor.df, pd.DataFrame)
    assert processor.df.shape[0] <= sample_data.shape[0]  # Preprocessing may reduce rows
    assert all(
        col not in processor.df.columns for col in config.preprocessing["drop_columns"]
    )  # Dropped columns are gone


# def test_aggregate_data(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
#     """Test the _aggregate_data method of DataProcessor.

#     This function verifies that the _aggregate_data method correctly performs
#     the aggregation based on the configuration.

#     :param sample_data: Input DataFrame containing sample data
#     :param config: Configuration object for the project
#     :param spark: SparkSession object
#     """
#     processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)

#     # Call the private method _aggregate_data
#     aggregated_data = processor._aggregate_data(sample_data)

#     # Verify the aggregation results
#     groupby_columns = config.preprocessing["aggregation_level"] + [config.preprocessing["date_column"]]
#     assert set(aggregated_data.columns) == set(
#         groupby_columns
#         + [config.preprocessing["target_column"]]
#         + [f"{col}_count" for col in config.preprocessing["aggregations"]["counts"]]
#     )
#     assert aggregated_data.shape[0] <= sample_data.shape[0]  # Aggregation reduces rows

#     # Check if counts are calculated correctly
#     for col in config.preprocessing["aggregations"]["counts"]:
#         assert f"{col}_count" in aggregated_data.columns


# def test_drop_short_series(sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession) -> None:
#     """Test the _drop_short_series method of DataProcessor.

#     This function verifies that the _drop_short_series method reduces the number of rows
#     when groups with fewer than the minimum required years are present.

#     :param sample_data: Input DataFrame containing sample data
#     :param config: Configuration object for the project
#     :param spark: SparkSession object
#     """
#     processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)

#     # Call the private method _drop_short_series
#     filtered_data = processor._drop_short_series(sample_data)

#     # Verify that the output has fewer rows than the input
#     assert len(filtered_data) <= len(sample_data)


def test_split_by_time_default_params(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the default parameters of the split_data method in DataProcessor.

    This function tests if the split_data method correctly splits the input DataFrame
    into train and test sets using default parameters.

    :param sample_data: Input DataFrame to be split
    :param config: Configuration object for the project
    :param spark: SparkSession object
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess_data()
    train, test = processor.split_by_time()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(processor.df)
    assert set(train.columns) == set(test.columns) == set(processor.df.columns)

    # # The following lines are just to mimick the behavior of delta tables in UC
    # # Just one time execution in order for all other tests to work
    train.to_csv((CATALOG_DIR / "train_set.csv").as_posix(), index=False)  # noqa
    test.to_csv((CATALOG_DIR / "test_set.csv").as_posix(), index=False)  # noqa


def test_preprocess_empty_dataframe(config: ProjectConfig, spark_session: SparkSession) -> None:
    """Test the preprocess method with an empty DataFrame.

    This function tests if the preprocess method correctly handles an empty DataFrame
    and raises KeyError.

    :param config: Configuration object for the project
    :param spark: SparkSession object
    :raises KeyError: If the preprocess method handles empty DataFrame correctly
    """
    processor = DataProcessor(pandas_df=pd.DataFrame([]), config=config, spark=spark_session)
    with pytest.raises(KeyError):
        processor.preprocess_data()


@pytest.mark.skip(reason="depends on delta tables on Databricks")
def test_save_to_catalog_succesfull(
    sample_data: pd.DataFrame, config: ProjectConfig, spark_session: SparkSession
) -> None:
    """Test the successful saving of data to the catalog.

    This function processes sample data, splits it into train and test sets, and saves them to the catalog.
    It then asserts that the saved tables exist in the catalog.

    :param sample_data: The sample data to be processed and saved
    :param config: Configuration object for the project
    :param spark: SparkSession object for interacting with Spark
    """
    processor = DataProcessor(pandas_df=sample_data, config=config, spark=spark_session)
    processor.preprocess_data()
    train_set, test_set = processor.split_by_time()
    processor.save_to_catalog(train_set, test_set)
    processor.enable_change_data_feed()

    # Assert
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.train_set")
    assert spark_session.catalog.tableExists(f"{config.catalog_name}.{config.schema_name}.test_set")
