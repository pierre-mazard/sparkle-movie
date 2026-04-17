from pathlib import Path
from typing import Dict, Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType, LongType, StringType, StructField, StructType

from sparkle_movie.config import settings


REQUIRED_FILES = ("ratings.csv", "movies.csv")
GROUPLENS_BASE_URL = "https://files.grouplens.org/datasets/movielens"

RATINGS_SCHEMA = StructType(
    [
        StructField("userId", IntegerType(), nullable=True),
        StructField("movieId", IntegerType(), nullable=True),
        StructField("rating", FloatType(), nullable=True),
        StructField("timestamp", LongType(), nullable=True),
    ]
)
MOVIES_SCHEMA = StructType(
    [
        StructField("movieId", IntegerType(), nullable=True),
        StructField("title", StringType(), nullable=True),
        StructField("genres", StringType(), nullable=True),
    ]
)


def _dataset_dir() -> Path:
    return Path(settings.movielens_data_dir)


def _has_required_files(dataset_dir: Path) -> bool:
    return all((dataset_dir / file_name).exists() for file_name in REQUIRED_FILES)


def ensure_movielens_dataset() -> Path:
    dataset_dir = _dataset_dir()
    if _has_required_files(dataset_dir):
        return dataset_dir

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_name = f"{settings.movielens_size}.zip"
    zip_path = dataset_dir.parent / zip_name
    url = f"{GROUPLENS_BASE_URL}/{zip_name}"

    print(f"Downloading MovieLens dataset from: {url}")
    urlretrieve(url, zip_path)

    with ZipFile(zip_path, "r") as archive:
        archive.extractall(dataset_dir.parent)

    if not _has_required_files(dataset_dir):
        raise FileNotFoundError(
            f"Dataset extracted but required files are missing in {dataset_dir.resolve()}. "
            f"Expected files: {', '.join(REQUIRED_FILES)}"
        )

    return dataset_dir


def load_raw_movielens_dataframes(spark: SparkSession, dataset_dir: Path) -> Tuple[DataFrame, DataFrame]:
    ratings_df = (
        spark.read.option("header", True)
        .schema(RATINGS_SCHEMA)
        .csv(str(dataset_dir / "ratings.csv"))
    )

    movies_df = (
        spark.read.option("header", True)
        .schema(MOVIES_SCHEMA)
        .csv(str(dataset_dir / "movies.csv"))
    )

    return ratings_df, movies_df


def clean_movielens_dataframes(raw_ratings_df: DataFrame, raw_movies_df: DataFrame) -> Tuple[DataFrame, DataFrame, Dict[str, int]]:
    report: Dict[str, int] = {}
    report["ratings_rows_raw"] = raw_ratings_df.count()
    report["movies_rows_raw"] = raw_movies_df.count()

    report["ratings_missing_core"] = raw_ratings_df.filter(
        col("userId").isNull() | col("movieId").isNull() | col("rating").isNull()
    ).count()
    report["movies_missing_core"] = raw_movies_df.filter(
        col("movieId").isNull() | col("title").isNull()
    ).count()

    report["ratings_duplicates"] = (
        raw_ratings_df.groupBy("userId", "movieId", "timestamp").count().filter(col("count") > 1).count()
    )
    report["movies_duplicates"] = raw_movies_df.groupBy("movieId").count().filter(col("count") > 1).count()

    report["ratings_out_of_range"] = raw_ratings_df.filter((col("rating") < 0.0) | (col("rating") > 5.0)).count()

    ratings_df = (
        raw_ratings_df
        .dropna(subset=["userId", "movieId", "rating"])
        .dropDuplicates(["userId", "movieId", "timestamp"])
        .filter((col("rating") >= 0.0) & (col("rating") <= 5.0))
    )

    movies_df = (
        raw_movies_df
        .dropna(subset=["movieId", "title"])
        .dropDuplicates(["movieId"])
    )

    report["ratings_rows_clean"] = ratings_df.count()
    report["movies_rows_clean"] = movies_df.count()

    return ratings_df, movies_df, report


def load_movielens_dataframes(spark: SparkSession, dataset_dir: Path) -> Tuple[DataFrame, DataFrame]:
    raw_ratings_df, raw_movies_df = load_raw_movielens_dataframes(spark, dataset_dir)
    ratings_df, movies_df, _ = clean_movielens_dataframes(raw_ratings_df, raw_movies_df)

    return ratings_df, movies_df
