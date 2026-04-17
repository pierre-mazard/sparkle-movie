from pathlib import Path
import shutil
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from sparkle_movie.config import settings
from sparkle_movie.movielens import (
    clean_movielens_dataframes,
    ensure_movielens_dataset,
    load_raw_movielens_dataframes,
)


sns.set_theme(style="whitegrid")


def _safe_float(value: object, digits: int = 4) -> object:
    if value is None:
        return None
    if isinstance(value, (int,)):
        return value
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _ensure_dirs() -> Dict[str, Path]:
    output_dir = Path(settings.output_dir)
    figures_dir = Path(settings.figures_dir)
    reports_dir = Path("reports")
    tableau_dir = output_dir / "tableau"
    cleaned_dir = output_dir / "cleaned"

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tableau_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "figures_dir": figures_dir,
        "reports_dir": reports_dir,
        "tableau_dir": tableau_dir,
        "cleaned_dir": cleaned_dir,
    }


def _rows_to_markdown(rows: List[Dict[str, object]], columns: Iterable[str]) -> str:
    columns = list(columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, separator] + body)


def _collect_preview(df: DataFrame, n: int = 10) -> List[Dict[str, object]]:
    rows = df.limit(n).toPandas().to_dict(orient="records")
    return rows


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _reset_output_path(path: Path) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _rating_summary(ratings_df: DataFrame) -> Dict[str, object]:
    row = (
        ratings_df.agg(
            F.count("*").alias("count"),
            F.avg("rating").alias("mean"),
            F.stddev_pop("rating").alias("stddev"),
            F.min("rating").alias("min"),
            F.expr("percentile_approx(rating, 0.25)").alias("q25"),
            F.expr("percentile_approx(rating, 0.5)").alias("median"),
            F.expr("percentile_approx(rating, 0.75)").alias("q75"),
            F.max("rating").alias("max"),
        )
        .first()
        .asDict()
    )
    return {k: _safe_float(v) for k, v in row.items()}


def _activity_summary(ratings_df: DataFrame, key_col: str) -> Dict[str, object]:
    activity = ratings_df.groupBy(key_col).agg(F.count("*").alias("num_ratings"))
    row = (
        activity.agg(
            F.count("*").alias("entities"),
            F.avg("num_ratings").alias("mean_ratings"),
            F.expr("percentile_approx(num_ratings, 0.5)").alias("median_ratings"),
            F.expr("percentile_approx(num_ratings, 0.9)").alias("p90_ratings"),
            F.max("num_ratings").alias("max_ratings"),
        )
        .first()
        .asDict()
    )
    return {k: _safe_float(v) for k, v in row.items()}


def run_full_analysis(spark: SparkSession) -> Dict[str, Path]:
    paths = _ensure_dirs()

    dataset_dir = ensure_movielens_dataset()
    raw_ratings_df, raw_movies_df = load_raw_movielens_dataframes(spark, dataset_dir)
    ratings_df, movies_df, cleaning_report = clean_movielens_dataframes(raw_ratings_df, raw_movies_df)

    ratings_df.cache()
    movies_df.cache()

    raw_ratings_preview = _collect_preview(raw_ratings_df, 10)
    raw_movies_preview = _collect_preview(raw_movies_df, 10)
    cleaned_ratings_preview = _collect_preview(ratings_df, 10)
    cleaned_movies_preview = _collect_preview(movies_df, 10)

    rating_summary = _rating_summary(ratings_df)
    user_activity_summary = _activity_summary(ratings_df, "userId")
    movie_activity_summary = _activity_summary(ratings_df, "movieId")

    movies_with_ratings = ratings_df.join(movies_df, on="movieId", how="inner")

    movie_stats = (
        movies_with_ratings.groupBy("movieId", "title")
        .agg(
            F.count("*").alias("rating_count"),
            F.avg("rating").alias("mean_rating"),
            F.expr("percentile_approx(rating, 0.5)").alias("median_rating"),
            F.stddev_pop("rating").alias("std_rating"),
        )
    )

    global_mean = float(rating_summary["mean"])
    min_votes = int(
        movie_stats.select(F.expr("percentile_approx(rating_count, 0.90)").alias("min_votes")).first()["min_votes"]
    )

    scored_movies = movie_stats.withColumn(
        "bayesian_score",
        ((F.col("rating_count") / (F.col("rating_count") + F.lit(min_votes))) * F.col("mean_rating"))
        + ((F.lit(min_votes) / (F.col("rating_count") + F.lit(min_votes))) * F.lit(global_mean)),
    )

    top_movies_mean_df = (
        scored_movies.filter(F.col("rating_count") >= 20)
        .orderBy(F.desc("mean_rating"), F.desc("rating_count"))
        .limit(20)
    )

    top_movies_bayesian_df = scored_movies.orderBy(F.desc("bayesian_score"), F.desc("rating_count")).limit(20)

    genres_df = (
        movies_df.select("movieId", F.explode(F.split(F.col("genres"), "\\|")).alias("genre"))
        .filter(F.col("genre") != "(no genres listed)")
    )

    genre_stats_df = (
        ratings_df.join(genres_df, on="movieId", how="inner")
        .groupBy("genre")
        .agg(
            F.count("*").alias("rating_count"),
            F.countDistinct("userId").alias("user_count"),
            F.countDistinct("movieId").alias("movie_count"),
            F.avg("rating").alias("mean_rating"),
            F.expr("percentile_approx(rating, 0.5)").alias("median_rating"),
        )
        .orderBy(F.desc("rating_count"))
    )

    top_genres_df = genre_stats_df.limit(20)

    top_movies_mean_pd = top_movies_mean_df.toPandas()
    top_movies_bayesian_pd = top_movies_bayesian_df.toPandas()
    top_genres_pd = top_genres_df.toPandas()
    genre_stats_pd = genre_stats_df.toPandas()

    ratings_distribution_pd = ratings_df.groupBy("rating").count().orderBy("rating").toPandas()

    top_movies_mean_pd = top_movies_mean_pd.rename(columns={"title": "movie_title"})
    top_movies_bayesian_pd = top_movies_bayesian_pd.rename(columns={"title": "movie_title"})

    for col in ["mean_rating", "median_rating", "std_rating", "bayesian_score"]:
        if col in top_movies_mean_pd.columns:
            top_movies_mean_pd[col] = top_movies_mean_pd[col].round(4)
        if col in top_movies_bayesian_pd.columns:
            top_movies_bayesian_pd[col] = top_movies_bayesian_pd[col].round(4)

    for col in ["mean_rating", "median_rating"]:
        if col in top_genres_pd.columns:
            top_genres_pd[col] = top_genres_pd[col].round(4)
        if col in genre_stats_pd.columns:
            genre_stats_pd[col] = genre_stats_pd[col].round(4)

    cleaned_ratings_out = paths["cleaned_dir"] / "ratings_cleaned.parquet"
    cleaned_movies_out = paths["cleaned_dir"] / "movies_cleaned.parquet"
    _reset_output_path(cleaned_ratings_out)
    _reset_output_path(cleaned_movies_out)
    ratings_df.toPandas().to_parquet(cleaned_ratings_out, index=False)
    movies_df.toPandas().to_parquet(cleaned_movies_out, index=False)

    tableau_top_movies = paths["tableau_dir"] / "top_movies_bayesian.csv"
    tableau_top_genres = paths["tableau_dir"] / "genre_popularity.csv"
    tableau_rating_dist = paths["tableau_dir"] / "rating_distribution.csv"
    tableau_movie_scores = paths["tableau_dir"] / "movie_scores.csv"

    top_movies_bayesian_pd.to_csv(tableau_top_movies, index=False)
    genre_stats_pd.to_csv(tableau_top_genres, index=False)
    ratings_distribution_pd.to_csv(tableau_rating_dist, index=False)
    scored_movies.orderBy(F.desc("bayesian_score")).toPandas().to_csv(tableau_movie_scores, index=False)

    fig_rating_dist = paths["figures_dir"] / "ratings_distribution.png"
    fig_top_movies = paths["figures_dir"] / "top_movies_bayesian.png"
    fig_top_genres = paths["figures_dir"] / "top_genres_popularity.png"
    fig_mean_vs_count = paths["figures_dir"] / "movie_mean_vs_count.png"

    plt.figure(figsize=(8, 5))
    sns.barplot(data=ratings_distribution_pd, x="rating", y="count", color="#3D5A80")
    plt.title("Ratings distribution")
    plt.xlabel("Rating value")
    plt.ylabel("Number of ratings")
    plt.tight_layout()
    plt.savefig(fig_rating_dist, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plot_movies = top_movies_bayesian_pd.head(10).sort_values("bayesian_score", ascending=True)
    sns.barplot(
        data=plot_movies,
        x="bayesian_score",
        y="movie_title",
        hue="movie_title",
        palette="viridis",
        legend=False,
    )
    plt.title("Top 10 movies by Bayesian score")
    plt.xlabel("Bayesian score")
    plt.ylabel("Movie")
    plt.tight_layout()
    plt.savefig(fig_top_movies, dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    plot_genres = top_genres_pd.head(10).sort_values("rating_count", ascending=True)
    sns.barplot(
        data=plot_genres,
        x="rating_count",
        y="genre",
        hue="genre",
        palette="mako",
        legend=False,
    )
    plt.title("Top 10 genres by popularity (rating count)")
    plt.xlabel("Number of ratings")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig(fig_top_genres, dpi=180)
    plt.close()

    scored_movies_pd = scored_movies.select("title", "rating_count", "mean_rating", "bayesian_score").toPandas()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=scored_movies_pd, x="rating_count", y="mean_rating", alpha=0.35, s=35)
    plt.title("Movie mean rating vs number of ratings")
    plt.xlabel("Rating count")
    plt.ylabel("Mean rating")
    plt.xscale("log")
    plt.tight_layout()
    plt.savefig(fig_mean_vs_count, dpi=180)
    plt.close()

    loading_report = (
        "# Loading Report\n\n"
        f"- Dataset directory: {dataset_dir.resolve()}\n"
        f"- Raw ratings rows: {cleaning_report['ratings_rows_raw']}\n"
        f"- Raw movies rows: {cleaning_report['movies_rows_raw']}\n"
        "\n## Raw ratings preview (10 rows)\n\n"
        + _rows_to_markdown(raw_ratings_preview, ["userId", "movieId", "rating", "timestamp"])
        + "\n\n## Raw movies preview (10 rows)\n\n"
        + _rows_to_markdown(raw_movies_preview, ["movieId", "title", "genres"])
        + "\n"
    )
    _write_text(paths["reports_dir"] / "loading_report.md", loading_report)

    cleaning_report_md = (
        "# Cleaning Report\n\n"
        f"- Ratings missing core fields: {cleaning_report['ratings_missing_core']}\n"
        f"- Movies missing core fields: {cleaning_report['movies_missing_core']}\n"
        f"- Ratings duplicate key groups: {cleaning_report['ratings_duplicates']}\n"
        f"- Movies duplicate movieId groups: {cleaning_report['movies_duplicates']}\n"
        f"- Ratings out of allowed range [0,5]: {cleaning_report['ratings_out_of_range']}\n"
        f"- Clean ratings rows: {cleaning_report['ratings_rows_clean']}\n"
        f"- Clean movies rows: {cleaning_report['movies_rows_clean']}\n"
        "\n## Clean ratings preview (10 rows)\n\n"
        + _rows_to_markdown(cleaned_ratings_preview, ["userId", "movieId", "rating", "timestamp"])
        + "\n\n## Clean movies preview (10 rows)\n\n"
        + _rows_to_markdown(cleaned_movies_preview, ["movieId", "title", "genres"])
        + "\n"
    )
    _write_text(paths["reports_dir"] / "cleaning_report.md", cleaning_report_md)

    exploration_report = (
        "# Exploration Report\n\n"
        "## Ratings distribution summary\n\n"
        f"- Count: {rating_summary['count']}\n"
        f"- Mean: {rating_summary['mean']}\n"
        f"- Median: {rating_summary['median']}\n"
        f"- Q1: {rating_summary['q25']}\n"
        f"- Q3: {rating_summary['q75']}\n"
        f"- Std dev: {rating_summary['stddev']}\n"
        f"- Min/Max: {rating_summary['min']} / {rating_summary['max']}\n"
        "\n## User activity summary\n\n"
        f"- Number of users: {user_activity_summary['entities']}\n"
        f"- Mean ratings per user: {user_activity_summary['mean_ratings']}\n"
        f"- Median ratings per user: {user_activity_summary['median_ratings']}\n"
        f"- P90 ratings per user: {user_activity_summary['p90_ratings']}\n"
        f"- Max ratings by one user: {user_activity_summary['max_ratings']}\n"
        "\n## Movie activity summary\n\n"
        f"- Number of movies with ratings: {movie_activity_summary['entities']}\n"
        f"- Mean ratings per movie: {movie_activity_summary['mean_ratings']}\n"
        f"- Median ratings per movie: {movie_activity_summary['median_ratings']}\n"
        f"- P90 ratings per movie: {movie_activity_summary['p90_ratings']}\n"
        f"- Max ratings for one movie: {movie_activity_summary['max_ratings']}\n"
    )
    _write_text(paths["reports_dir"] / "exploration_report.md", exploration_report)

    analysis_report = (
        "# Analysis Report\n\n"
        "## Methodological notes\n\n"
        f"- Global mean rating: {round(global_mean, 4)}\n"
        f"- Bayesian minimum vote prior (90th percentile of vote counts): {min_votes}\n"
        "\n## Best rated movies by mean score (minimum 20 ratings)\n\n"
        + _rows_to_markdown(
            top_movies_mean_pd.head(10).to_dict(orient="records"),
            ["movie_title", "rating_count", "mean_rating", "median_rating", "std_rating", "bayesian_score"],
        )
        + "\n\n## Best rated movies by Bayesian score\n\n"
        + _rows_to_markdown(
            top_movies_bayesian_pd.head(10).to_dict(orient="records"),
            ["movie_title", "rating_count", "mean_rating", "median_rating", "std_rating", "bayesian_score"],
        )
        + "\n\n## Most popular genres\n\n"
        + _rows_to_markdown(
            top_genres_pd.head(10).to_dict(orient="records"),
            ["genre", "rating_count", "user_count", "movie_count", "mean_rating", "median_rating"],
        )
        + "\n"
    )
    _write_text(paths["reports_dir"] / "analysis_report.md", analysis_report)

    tableau_guide = (
        "# Tableau Guide\n\n"
        "Use the CSV outputs under data/processed/tableau.\n\n"
        "## Recommended sheets\n\n"
        "1. Top movies by Bayesian score\n"
        "- Source: top_movies_bayesian.csv\n"
        "- Dimensions: movie_title\n"
        "- Measures: bayesian_score, rating_count, mean_rating\n\n"
        "2. Genre popularity\n"
        "- Source: genre_popularity.csv\n"
        "- Dimensions: genre\n"
        "- Measures: rating_count, user_count, mean_rating, median_rating\n\n"
        "3. Rating distribution\n"
        "- Source: rating_distribution.csv\n"
        "- Dimensions: rating\n"
        "- Measure: count\n"
    )
    _write_text(paths["reports_dir"] / "tableau_guide.md", tableau_guide)

    summary_report = (
        "# Pipeline Summary\n\n"
        "Generated artifacts:\n\n"
        f"- Reports: {paths['reports_dir'].resolve()}\n"
        f"- Figures: {paths['figures_dir'].resolve()}\n"
        f"- Tableau CSV: {paths['tableau_dir'].resolve()}\n"
        f"- Clean parquet: {paths['cleaned_dir'].resolve()}\n"
    )
    _write_text(paths["reports_dir"] / "pipeline_summary.md", summary_report)

    ratings_df.unpersist()
    movies_df.unpersist()

    return {
        "reports_dir": paths["reports_dir"],
        "figures_dir": paths["figures_dir"],
        "tableau_dir": paths["tableau_dir"],
        "cleaned_dir": paths["cleaned_dir"],
    }
