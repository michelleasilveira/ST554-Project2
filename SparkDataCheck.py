# Creating Python code from colab
"""
SparkDataCheck.py
-----------------
A data quality class that wraps a Spark SQL DataFrame and provides
methods for validating and summarizing data.

Author: Michelle A Silveira
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from functools import reduce
from pyspark.sql.types import *
import pandas as pd


class SparkDataCheck:
    """
    A class that wraps a Spark SQL DataFrame and provides functionality
    for cleaning and checking data quality.

    Attributes
    ----------
    df : pyspark.sql.DataFrame
        The underlying Spark SQL DataFrame.
    """

    # Column types recognized as numeric
    _NUMERIC_TYPES = {
        "float", "int", "long", "bigint", "double", "integer",
        "smallint", "tinyint", "short", "longint"
    }

    def __init__(self, dataframe):
        """
        Initialize SparkDataCheck with a Spark DataFrame.

        Parameters
        ----------
        dataframe : pyspark.sql.DataFrame
            The Spark SQL DataFrame to wrap.
        """
        self.df = dataframe

    @classmethod
    def from_csv(cls, spark, path):
        """
        Create a SparkDataCheck instance by reading a CSV file.

        Parameters
        ----------
        spark : SparkSession
            The active Spark session.
        path : str
            Path to the CSV file (local or HDFS).

        Returns
        -------
        SparkDataCheck
            A new SparkDataCheck instance wrapping the loaded DataFrame.
        """
        df = spark.read.load(
            path,
            format="csv",
            header=True,
            inferSchema=True
        )
        return cls(df)

    @classmethod
    def from_pandas(cls, spark, pandas_df):
        """
        Create a SparkDataCheck instance from a standard pandas DataFrame.

        Parameters
        ----------
        spark : SparkSession
            The active Spark session.
        pandas_df : pandas.DataFrame
            A standard (non-Spark) pandas DataFrame.

        Returns
        -------
        SparkDataCheck
            A new SparkDataCheck instance wrapping the converted DataFrame.
        """
        df = spark.createDataFrame(pandas_df)
        return cls(df)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _is_numeric(self, col_name):
        """Return True if the column is a numeric type."""
        col_types = dict(self.df.dtypes)
        if col_name not in col_types:
            return False
        dtype = col_types[col_name].lower()
        # Handle decimal(precision, scale)
        if dtype.startswith("decimal"):
            return True
        return dtype in self._NUMERIC_TYPES

    def _is_string(self, col_name):
        """Return True if the column is a string type."""
        col_types = dict(self.df.dtypes)
        if col_name not in col_types:
            return False
        return col_types[col_name].lower() == "string"

    def _get_spark_column(self, col_name):
        """Helper to get a Spark column, properly quoting if necessary."""
        # Always quote column names to safely handle special characters like dots, parentheses, etc.
        return F.col(f"`{col_name}`")

    # ------------------------------------------------------------------ #
    #  Validation methods (return self for chaining)                      #
    # ------------------------------------------------------------------ #

    def check_numeric_bounds(self, col_name, lower=None, upper=None):
        """
        Check whether each value in a numeric column falls within
        user-defined bounds (inclusive).  Appends a boolean column named
        ``<col_name>_in_bounds`` to the DataFrame.  NULL inputs produce
        NULL outputs.

        Parameters
        ----------
        col_name : str
            The numeric column to validate.
        lower : numeric, optional
            Inclusive lower bound.  At least one of lower/upper must be given.
        upper : numeric, optional
            Inclusive upper bound.  At least one of lower/upper must be given.

        Returns
        -------
        self : SparkDataCheck
            Returns itself (chainable).
        """
        if lower is None and upper is None:
            print("Error: At least one of 'lower' or 'upper' must be provided.")
            return self

        if not self._is_numeric(col_name):
            print(
                f"Column '{col_name}' is not numeric "
                f"(type: {dict(self.df.dtypes).get(col_name, 'unknown')}). "
                "No modification made."
            )
            return self

        result_col = f"{col_name}_in_bounds"
        col = self._get_spark_column(col_name)

        if lower is not None and upper is not None:
            check_expr = F.when(col.isNull(), None).otherwise(col.between(lower, upper))
        elif lower is not None:
            check_expr = F.when(col.isNull(), None).otherwise(col >= lower)
        else:
            check_expr = F.when(col.isNull(), None).otherwise(col <= upper)

        self.df = self.df.withColumn(result_col, check_expr)
        return self

    def check_string_levels(self, col_name, levels):
        """
        Check whether each value in a string column belongs to a set of
        valid levels.  Appends a boolean column named
        ``<col_name>_valid_level`` to the DataFrame.  NULL inputs produce
        NULL outputs.

        Parameters
        ----------
        col_name : str
            The string column to validate.
        levels : list of str
            The allowed values for the column.

        Returns
        -------
        self : SparkDataCheck
            Returns itself (chainable).
        """
        if not self._is_string(col_name):
            print(
                f"Column '{col_name}' is not a string column "
                f"(type: {dict(self.df.dtypes).get(col_name, 'unknown')}). "
                "No modification made."
            )
            return self

        result_col = f"{col_name}_valid_level"
        col = self._get_spark_column(col_name)

        check_expr = F.when(col.isNull(), None).otherwise(col.isin(levels))
        self.df = self.df.withColumn(result_col, check_expr)
        return self

    def check_missing(self, col_name):
        """
        Check whether each value in a column is NULL.  Appends a boolean
        column named ``<col_name>_is_missing`` to the DataFrame.

        Parameters
        ----------
        col_name : str
            The column to check for missing values.

        Returns
        -------
        self : SparkDataCheck
            Returns itself (chainable).
        """
        result_col = f"{col_name}_is_missing"
        self.df = self.df.withColumn(result_col, self._get_spark_column(col_name).isNull())
        return self

    # ------------------------------------------------------------------ #
    #  Summarization methods (return a pandas DataFrame)                  #
    # ------------------------------------------------------------------ #

    def numeric_summary(self, col_name=None, group_by=None):
        """
        Report the min and max of a numeric column (or all numeric columns).
        Returns a standard pandas DataFrame.

        Parameters
        ----------
        col_name : str, optional
            The numeric column to summarize.  If None, all numeric columns
            in the DataFrame are summarized.
        group_by : str, optional
            An optional column to group results by.

        Returns
        -------
        pandas.DataFrame or None
            A pandas DataFrame containing min/max values, or None if the
            specified column is not numeric.
        """
        if col_name is not None:
            # --- single column ---
            if not self._is_numeric(col_name):
                print(f"Column '{col_name}' is not numeric.")
                return None

            agg_exprs = [
                F.min(self._get_spark_column(col_name)).alias(f"{col_name}_min"),
                F.max(self._get_spark_column(col_name)).alias(f"{col_name}_max"),
            ]
            if group_by is not None:
                result = self.df.groupBy(self._get_spark_column(group_by)).agg(*agg_exprs).orderBy(self._get_spark_column(group_by))
            else:
                result = self.df.agg(*agg_exprs)

            return result.toPandas()

        else:
            # --- all numeric columns ---
            numeric_cols = [
                c for c, t in self.df.dtypes
                if t.lower() in self._NUMERIC_TYPES or t.lower().startswith("decimal")
            ]

            if not numeric_cols:
                return pd.DataFrame()

            if group_by is not None:
                # Compute per-column and merge on group_by
                partials = []
                for c in numeric_cols:
                    agg_exprs = [
                        F.min(self._get_spark_column(c)).alias(f"{c}_min"),
                        F.max(self._get_spark_column(c)).alias(f"{c}_max"),
                    ]
                    part = (
                        self.df.groupBy(self._get_spark_column(group_by))
                        .agg(*agg_exprs)
                        .orderBy(self._get_spark_column(group_by))
                        .toPandas()
                    )
                    partials.append(part)
                combined = reduce(
                    lambda left, right: pd.merge(left, right, on=group_by), partials
                )
            else:
                agg_exprs = []
                for c in numeric_cols:
                    agg_exprs.append(F.min(self._get_spark_column(c)).alias(f"{c}_min"))
                    agg_exprs.append(F.max(self._get_spark_column(c)).alias(f"{c}_max"))
                combined = self.df.agg(*agg_exprs).toPandas()

            return combined

    def string_counts(self, col1, col2=None):
        """
        Report value counts for one or two string columns.
        Returns a standard pandas DataFrame.

        Parameters
        ----------
        col1 : str
            The required string column.
        col2 : str, optional
            An optional second string column for cross-tabulation.

        Returns
        -------
        pandas.DataFrame or None
            A pandas DataFrame of counts, or None if any column is not a string.
        """
        if not self._is_string(col1):
            print(f"Column '{col1}' is not a string column.")
            return None

        if col2 is not None:
            if not self._is_string(col2):
                print(f"Column '{col2}' is not a string column.")
                return None
            group_cols = [self._get_spark_column(col1), self._get_spark_column(col2)]
        else:
            group_cols = [self._get_spark_column(col1)]

        result = (
            self.df
            .groupBy(*group_cols)
            .count()
            .orderBy(*group_cols)
            .toPandas()
        )
        return result
