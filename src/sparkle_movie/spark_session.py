from pyspark.sql import SparkSession
import os
import sys

from sparkle_movie.config import settings


def create_spark_session() -> SparkSession:
    python_exec = sys.executable
    os.environ.setdefault("PYSPARK_PYTHON", python_exec)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", python_exec)

    return (
        SparkSession.builder.appName(settings.app_name)
        .master(settings.spark_master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.pyspark.python", python_exec)
        .config("spark.pyspark.driver.python", python_exec)
        .getOrCreate()
    )
