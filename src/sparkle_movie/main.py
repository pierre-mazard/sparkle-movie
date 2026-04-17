import argparse
from pathlib import Path

from sparkle_movie.analysis import run_full_analysis
from sparkle_movie.config import settings
from sparkle_movie.modeling import run_modeling_pipeline
from sparkle_movie.movielens import ensure_movielens_dataset, load_movielens_dataframes
from sparkle_movie.spark_session import create_spark_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sparkle-movie command line")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Download MovieLens if needed, then load and preview ratings/movies data.",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run full EDA, cleaning report, trend analysis, visualizations, and Tableau exports.",
    )
    parser.add_argument(
        "--run-modeling",
        action="store_true",
        help="Run ALS, content-based, and user-KNN recommenders with comparative evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spark = create_spark_session()
    try:
        print(f"Spark session started: {spark.version}")
        print(f"MovieLens data directory: {Path(settings.movielens_data_dir).resolve()}")

        if args.prepare_data:
            dataset_dir = ensure_movielens_dataset()
            ratings_df, movies_df = load_movielens_dataframes(spark, dataset_dir)

            print(f"Dataset ready in: {dataset_dir.resolve()}")
            print(f"Ratings count: {ratings_df.count()}")
            print(f"Movies count: {movies_df.count()}")
            print("First 10 rows from ratings.csv")
            ratings_df.show(10, truncate=False)
            print("First 10 rows from movies.csv")
            movies_df.show(10, truncate=False)

        if args.run_analysis:
            outputs = run_full_analysis(spark)
            print("Full analysis pipeline completed.")
            for name, path in outputs.items():
                print(f"{name}: {path.resolve()}")

        if args.run_modeling:
            outputs = run_modeling_pipeline(spark)
            print("Modeling pipeline completed.")
            for name, path in outputs.items():
                print(f"{name}: {path.resolve()}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
