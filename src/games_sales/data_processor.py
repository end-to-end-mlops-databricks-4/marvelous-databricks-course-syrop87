import pandas as pd
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from games_sales.config import ProjectConfig


class DataProcessor:
    """Class to handle data preprocessing and saving to Databricks tables."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession | None = None) -> None:
        self.config = config
        self.df_raw = pandas_df
        self.spark = spark

    def preprocess_data(self) -> None:
        """Preprocess the data according to the configuration.

        :param df: Input DataFrame
        :return: Preprocessed DataFrame
        """
        df = self.df_raw.copy()
        df = df.drop(columns=self.config.preprocessing["drop_columns"])
        logger.info(f"Dropped columns: {self.config.preprocessing['drop_columns']}")

        df_agg = self._aggregate_data(df)
        logger.info(f"Aggregated data shape: {df_agg.shape}")

        df_agg = df_agg.dropna()
        logger.info(f"Data shape after dropping rows with NAs: {df_agg.shape}")

        df_agg = self._drop_short_series(df_agg)
        logger.info(f"Data shape after dropping short series: {df_agg.shape}")

        df_agg["Id"] = self._create_id(df_agg)
        logger.info("Created ID column by merging aggregation level columns and date column")

        df_agg = self._adjust_columns_types(df_agg)
        logger.info("Adjusted columns types")

        self.df = df_agg

    def _create_id(self, df: pd.DataFrame) -> pd.Series:
        """Create a unique ID by combining aggregation level columns and date column.

        :param df: Input DataFrame
        :return: Series with string IDs
        """
        agg_values = df[self.config.preprocessing["aggregation_level"]].astype(str).values
        date_values = df[self.config.preprocessing["date_column"]].astype(int).astype(str).values

        # Combine all values with underscore separator
        id_series = pd.Series(
            ["_".join(row) + "_" + date for row, date in zip(agg_values, date_values, strict=True)], index=df.index
        )
        return id_series

    def _adjust_columns_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adjust data types based on configuration settings.

        :param df: Input DataFrame
        :return: DataFrame with adjusted types
        """
        num_features = self.config.num_features + [self.config.target_column]
        for col in num_features:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        return df

    def _aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data based on configuration settings.

        :param df: Input DataFrame
        :return: Aggregated DataFrame
        """
        groupby_columns = self.config.preprocessing["aggregation_level"] + [self.config.preprocessing["date_column"]]
        df_grouped = df.groupby(groupby_columns, as_index=False)
        df_agg = df_grouped[self.config.preprocessing["target_column"]].sum()

        for aggregation in self.config.preprocessing["aggregations"]:
            if aggregation == "counts":
                columns_mapping = {
                    col: f"{col}_count" for col in self.config.preprocessing["aggregations"][aggregation]
                }
                df_agg_X = (
                    df_grouped[self.config.preprocessing["aggregations"][aggregation]]
                    .count()
                    .rename(columns=columns_mapping)
                )
                df_agg = df_agg.merge(df_agg_X, on=groupby_columns, how="left", validate="one_to_one")
            else:
                error_message = f"Aggregation {aggregation} not implemented"
                logger.error(error_message)
                raise NotImplementedError(error_message)
        return df_agg

    def _drop_short_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop series with fewer than the minimum required years.

        :param df: Input DataFrame
        :return: Filtered DataFrame
        """
        years_counts = df.groupby(self.config.preprocessing["aggregation_level"])[
            self.config.preprocessing["date_column"]
        ].transform("count")
        df_filtered = df[years_counts > self.config.preprocessing["min_years"]]
        logger.info(f"Dropped short series. Remaining shape: {df_filtered.shape}")
        return df_filtered

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

    def split_by_time(self, test_periods: int = 12, min_train_periods: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train and test sets.

        Takes k last periods for test for each group. for each group if enough data points are available.

        Args:
            test_periods: Number of periods for test set
            min_train_periods: Minimum number of periods required in training set

        """
        train_indices = []
        test_indices = []

        groupby_cols = self.config.preprocessing["aggregation_level"]
        date_col = self.config.preprocessing["date_column"]

        for name, group in self.df.groupby(groupby_cols):
            sorted_group = group.sort_values(date_col)
            total_periods = len(sorted_group)

            if total_periods >= test_periods + min_train_periods:
                split_idx = -test_periods
                train_indices.extend(sorted_group.index[:split_idx])
                test_indices.extend(sorted_group.index[split_idx:])
            else:
                logger.warning(f"Group {name} has only {total_periods} periods. Keeping all in train set.")
                train_indices.extend(sorted_group.index)

        train_df = self.df.loc[train_indices]
        test_df = self.df.loc[test_indices]

        logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
        return train_df, test_df
