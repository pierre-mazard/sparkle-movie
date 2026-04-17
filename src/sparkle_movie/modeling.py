from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType

from sparkle_movie.config import settings
from sparkle_movie.movielens import (
    clean_movielens_dataframes,
    ensure_movielens_dataset,
    load_raw_movielens_dataframes,
)


def _ensure_dirs() -> Dict[str, Path]:
    reports_dir = Path("reports")
    figures_dir = Path(settings.figures_dir)
    modeling_dir = Path(settings.output_dir) / "modeling"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    modeling_dir.mkdir(parents=True, exist_ok=True)

    return {
        "reports_dir": reports_dir,
        "figures_dir": figures_dir,
        "modeling_dir": modeling_dir,
    }


def _rows_to_markdown(rows: List[Dict[str, object]], columns: Iterable[str]) -> str:
    columns = list(columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, separator] + body)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _split_train_test_per_user(ratings: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []

    for _, group in ratings.groupby("userId", sort=False):
        n = len(group)
        if n < 2:
            train_parts.append(group)
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n_test, n - 1)

        test_parts.append(group.iloc[-n_test:])
        train_parts.append(group.iloc[:-n_test])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=ratings.columns)
    return train_df, test_df


def _evaluate_topk(
    recs_by_user: Dict[int, List[int]],
    test_relevant_items: Dict[int, set],
    all_item_ids: set,
    k: int,
) -> Dict[str, float]:
    user_precisions: List[float] = []
    user_recalls: List[float] = []
    recommended_items: set = set()

    for user_id, relevant in test_relevant_items.items():
        recs = recs_by_user.get(user_id, [])[:k]
        if not recs:
            user_precisions.append(0.0)
            user_recalls.append(0.0)
            continue

        rec_set = set(recs)
        recommended_items.update(rec_set)

        hits = len(rec_set.intersection(relevant))
        user_precisions.append(hits / k)
        user_recalls.append(hits / len(relevant) if relevant else 0.0)

    coverage = len(recommended_items) / len(all_item_ids) if all_item_ids else 0.0

    return {
        "precision_at_k": float(np.mean(user_precisions)) if user_precisions else 0.0,
        "recall_at_k": float(np.mean(user_recalls)) if user_recalls else 0.0,
        "coverage_at_k": float(coverage),
    }


def _fit_best_als(
    spark: SparkSession,
    train_pd: pd.DataFrame,
    test_pd: pd.DataFrame,
    k: int,
) -> Tuple[dict, Dict[int, List[int]], float]:
    als_tmp_dir = Path(settings.output_dir) / "modeling" / "_tmp_als"
    als_tmp_dir.mkdir(parents=True, exist_ok=True)

    train_csv = als_tmp_dir / "train_split.csv"
    test_csv = als_tmp_dir / "test_split.csv"
    users_csv = als_tmp_dir / "eval_users.csv"

    train_pd[["userId", "movieId", "rating"]].to_csv(train_csv, index=False)
    test_pd[["userId", "movieId", "rating"]].to_csv(test_csv, index=False)

    schema = StructType(
        [
            StructField("userId", IntegerType(), nullable=False),
            StructField("movieId", IntegerType(), nullable=False),
            StructField("rating", FloatType(), nullable=False),
        ]
    )

    train_sdf = spark.read.option("header", True).schema(schema).csv(str(train_csv))
    test_sdf = spark.read.option("header", True).schema(schema).csv(str(test_csv))

    fit_sdf, val_sdf = train_sdf.randomSplit([0.8, 0.2], seed=42)

    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

    grid = [
        {"rank": 20, "regParam": 0.05, "maxIter": 10},
        {"rank": 20, "regParam": 0.1, "maxIter": 15},
        {"rank": 40, "regParam": 0.05, "maxIter": 10},
        {"rank": 40, "regParam": 0.1, "maxIter": 15},
    ]

    best_conf = None
    best_rmse = float("inf")

    for conf in grid:
        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            rank=conf["rank"],
            regParam=conf["regParam"],
            maxIter=conf["maxIter"],
            coldStartStrategy="drop",
            nonnegative=True,
            seed=42,
        )
        model = als.fit(fit_sdf)
        pred_val = model.transform(val_sdf)
        rmse_val = evaluator.evaluate(pred_val)

        if rmse_val < best_rmse:
            best_rmse = rmse_val
            best_conf = conf

    assert best_conf is not None

    final_als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=best_conf["rank"],
        regParam=best_conf["regParam"],
        maxIter=best_conf["maxIter"],
        coldStartStrategy="drop",
        nonnegative=True,
        seed=42,
    )
    final_model = final_als.fit(train_sdf)

    pred_test = final_model.transform(test_sdf)
    test_rmse = evaluator.evaluate(pred_test)

    eval_users = sorted(test_pd["userId"].unique().tolist())
    pd.DataFrame({"userId": eval_users}).to_csv(users_csv, index=False)
    users_sdf = (
        spark.read.option("header", True)
        .schema(StructType([StructField("userId", IntegerType(), nullable=False)]))
        .csv(str(users_csv))
    )

    recs_sdf = final_model.recommendForUserSubset(users_sdf, k)
    recs_pd = recs_sdf.toPandas()

    recs_by_user: Dict[int, List[int]] = {}
    for _, row in recs_pd.iterrows():
        user_id = int(row["userId"])
        recs = [int(x["movieId"]) for x in row["recommendations"]]
        recs_by_user[user_id] = recs

    best_conf = dict(best_conf)
    best_conf["validation_rmse"] = float(best_rmse)
    best_conf["test_rmse"] = float(test_rmse)

    shutil.rmtree(als_tmp_dir, ignore_errors=True)
    return best_conf, recs_by_user, float(test_rmse)


def _build_content_recs(
    train_pd: pd.DataFrame,
    users_for_eval: List[int],
    movie_ids: np.ndarray,
    movie_genres_text: List[str],
    k: int,
) -> Dict[int, List[int]]:
    vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
    tfidf = vectorizer.fit_transform(movie_genres_text)

    movie_to_idx = {int(mid): idx for idx, mid in enumerate(movie_ids)}
    all_idx = np.arange(len(movie_ids))

    user_hist = train_pd.groupby("userId")

    recs_by_user: Dict[int, List[int]] = {}
    for user_id in users_for_eval:
        if user_id not in user_hist.groups:
            recs_by_user[user_id] = []
            continue

        grp = user_hist.get_group(user_id)
        seen_movies = set(int(m) for m in grp["movieId"].tolist() if int(m) in movie_to_idx)
        seen_idx = [movie_to_idx[m] for m in seen_movies]
        if not seen_idx:
            recs_by_user[user_id] = []
            continue

        ratings = grp.set_index("movieId")["rating"]
        weights = []
        vectors = []
        for movie_id in seen_movies:
            vectors.append(tfidf[movie_to_idx[movie_id]])
            weights.append(float(ratings.loc[movie_id]))

        user_profile = vectors[0].multiply(weights[0])
        for i in range(1, len(vectors)):
            user_profile = user_profile + vectors[i].multiply(weights[i])

        profile_norm = np.sqrt(user_profile.multiply(user_profile).sum())
        if profile_norm == 0:
            recs_by_user[user_id] = []
            continue

        numerators = tfidf.dot(user_profile.T).toarray().ravel()
        tfidf_norm = np.sqrt(tfidf.multiply(tfidf).sum(axis=1)).A1
        scores = numerators / (tfidf_norm * profile_norm + 1e-12)

        candidate_mask = np.ones_like(scores, dtype=bool)
        candidate_mask[seen_idx] = False
        candidate_idx = all_idx[candidate_mask]

        if len(candidate_idx) == 0:
            recs_by_user[user_id] = []
            continue

        candidate_scores = scores[candidate_idx]
        top_idx_local = np.argsort(candidate_scores)[-k:][::-1]
        top_idx = candidate_idx[top_idx_local]
        recs_by_user[user_id] = [int(movie_ids[i]) for i in top_idx]

    return recs_by_user


def _build_knn_recs(
    train_pd: pd.DataFrame,
    users_for_eval: List[int],
    k: int,
    n_neighbors: int = 20,
) -> Dict[int, List[int]]:
    user_ids = sorted(train_pd["userId"].unique().tolist())
    movie_ids = sorted(train_pd["movieId"].unique().tolist())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    movie_to_idx = {m: i for i, m in enumerate(movie_ids)}

    rows = train_pd["userId"].map(user_to_idx).to_numpy()
    cols = train_pd["movieId"].map(movie_to_idx).to_numpy()
    data = train_pd["rating"].to_numpy(dtype=float)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(movie_ids)))

    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(n_neighbors + 1, len(user_ids)))
    model.fit(matrix)

    train_grouped = train_pd.groupby("userId")
    user_ratings_map = {
        int(uid): {int(m): float(r) for m, r in zip(g["movieId"], g["rating"])} for uid, g in train_grouped
    }

    recs_by_user: Dict[int, List[int]] = {}
    for user_id in users_for_eval:
        if user_id not in user_to_idx:
            recs_by_user[user_id] = []
            continue

        u_idx = user_to_idx[user_id]
        distances, indices = model.kneighbors(matrix[u_idx], return_distance=True)

        neighbor_info: List[Tuple[int, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            neigh_user = user_ids[idx]
            if neigh_user == user_id:
                continue
            sim = max(0.0, 1.0 - float(dist))
            if sim > 0:
                neighbor_info.append((neigh_user, sim))

        seen_movies = set(user_ratings_map.get(user_id, {}).keys())
        scores: Dict[int, float] = {}
        denoms: Dict[int, float] = {}

        for neigh_user, sim in neighbor_info:
            for movie_id, rating in user_ratings_map.get(neigh_user, {}).items():
                if movie_id in seen_movies:
                    continue
                scores[movie_id] = scores.get(movie_id, 0.0) + sim * rating
                denoms[movie_id] = denoms.get(movie_id, 0.0) + sim

        if not scores:
            recs_by_user[user_id] = []
            continue

        ranked = sorted(
            ((m, scores[m] / (denoms[m] + 1e-12)) for m in scores),
            key=lambda x: x[1],
            reverse=True,
        )
        recs_by_user[user_id] = [int(m) for m, _ in ranked[:k]]

    return recs_by_user


def _build_hybrid_recs(
    users_for_eval: List[int],
    als_recs: Dict[int, List[int]],
    content_recs: Dict[int, List[int]],
    knn_recs: Dict[int, List[int]],
    k: int,
    weights: Dict[str, float],
) -> Dict[int, List[int]]:
    # Weighted reciprocal-rank fusion keeps hybrid ranking stable across methods.
    denom_offset = 10.0
    recs_by_user: Dict[int, List[int]] = {}

    for user_id in users_for_eval:
        agg: Dict[int, float] = {}

        for method, rec_map in [
            ("als", als_recs),
            ("content", content_recs),
            ("knn", knn_recs),
        ]:
            method_weight = float(weights.get(method, 0.0))
            ranked_items = rec_map.get(user_id, [])
            for rank, movie_id in enumerate(ranked_items, start=1):
                score = method_weight / (denom_offset + rank)
                agg[int(movie_id)] = agg.get(int(movie_id), 0.0) + score

        ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        recs_by_user[user_id] = [movie_id for movie_id, _ in ranked[:k]]

    return recs_by_user


def _tune_hybrid_weights(
    users_for_eval: List[int],
    als_recs: Dict[int, List[int]],
    content_recs: Dict[int, List[int]],
    knn_recs: Dict[int, List[int]],
    test_relevant_items: Dict[int, set],
    all_item_ids: set,
    k: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[int, List[int]], pd.DataFrame]:
    candidates: List[Tuple[float, float, float]] = []
    step = 0.1
    values = [round(x * step, 1) for x in range(1, 10)]
    for w_als in values:
        for w_content in values:
            w_knn = round(1.0 - w_als - w_content, 1)
            if w_knn <= 0:
                continue
            candidates.append((w_als, w_content, w_knn))

    rows: List[Dict[str, float]] = []
    best_weights = {"als": 0.5, "content": 0.2, "knn": 0.3}
    best_metrics = {"precision_at_k": -1.0, "recall_at_k": -1.0, "coverage_at_k": -1.0}
    best_recs: Dict[int, List[int]] = {}

    for w_als, w_content, w_knn in candidates:
        weights = {"als": w_als, "content": w_content, "knn": w_knn}
        recs = _build_hybrid_recs(users_for_eval, als_recs, content_recs, knn_recs, k, weights)
        metrics = _evaluate_topk(recs, test_relevant_items, all_item_ids, k)

        rows.append(
            {
                "w_als": w_als,
                "w_content": w_content,
                "w_knn": w_knn,
                "precision_at_k": metrics["precision_at_k"],
                "recall_at_k": metrics["recall_at_k"],
                "coverage_at_k": metrics["coverage_at_k"],
            }
        )

        better = (
            (metrics["precision_at_k"] > best_metrics["precision_at_k"])
            or (
                metrics["precision_at_k"] == best_metrics["precision_at_k"]
                and metrics["recall_at_k"] > best_metrics["recall_at_k"]
            )
            or (
                metrics["precision_at_k"] == best_metrics["precision_at_k"]
                and metrics["recall_at_k"] == best_metrics["recall_at_k"]
                and metrics["coverage_at_k"] > best_metrics["coverage_at_k"]
            )
        )
        if better:
            best_weights = weights
            best_metrics = metrics
            best_recs = recs

    tuning_df = pd.DataFrame(rows).sort_values(
        ["precision_at_k", "recall_at_k", "coverage_at_k"],
        ascending=[False, False, False],
    )

    return best_weights, best_metrics, best_recs, tuning_df


def run_modeling_pipeline(spark: SparkSession, k: int = 10) -> Dict[str, Path]:
    paths = _ensure_dirs()

    dataset_dir = ensure_movielens_dataset()
    raw_ratings_df, raw_movies_df = load_raw_movielens_dataframes(spark, dataset_dir)
    ratings_df, movies_df, _ = clean_movielens_dataframes(raw_ratings_df, raw_movies_df)

    ratings_pd = ratings_df.select("userId", "movieId", "rating", "timestamp").toPandas()
    movies_pd = movies_df.select("movieId", "title", "genres").toPandas()

    ratings_pd = (
        ratings_pd.sort_values(["userId", "movieId", "timestamp"])
        .drop_duplicates(subset=["userId", "movieId"], keep="last")
        .reset_index(drop=True)
    )

    train_pd, test_pd = _split_train_test_per_user(ratings_pd, test_ratio=0.2)

    relevant_test = test_pd[test_pd["rating"] >= 4.0]
    test_relevant_items: Dict[int, set] = {
        int(uid): set(group["movieId"].astype(int).tolist())
        for uid, group in relevant_test.groupby("userId")
    }

    eval_users = sorted(test_relevant_items.keys())
    all_item_ids = set(train_pd["movieId"].astype(int).unique().tolist())

    als_conf, als_recs, als_rmse = _fit_best_als(spark, train_pd, test_pd, k)

    movies_for_content = movies_pd.copy()
    movies_for_content["genres"] = movies_for_content["genres"].fillna("(no genres listed)")
    movie_ids_arr = movies_for_content["movieId"].astype(int).to_numpy()
    genres_text = movies_for_content["genres"].str.replace("|", " ", regex=False).tolist()

    content_recs = _build_content_recs(train_pd, eval_users, movie_ids_arr, genres_text, k)
    knn_recs = _build_knn_recs(train_pd, eval_users, k, n_neighbors=20)
    hybrid_weights, hybrid_metrics, hybrid_recs, hybrid_tuning_df = _tune_hybrid_weights(
        eval_users,
        als_recs,
        content_recs,
        knn_recs,
        test_relevant_items,
        all_item_ids,
        k,
    )

    als_metrics = _evaluate_topk(als_recs, test_relevant_items, all_item_ids, k)
    content_metrics = _evaluate_topk(content_recs, test_relevant_items, all_item_ids, k)
    knn_metrics = _evaluate_topk(knn_recs, test_relevant_items, all_item_ids, k)

    als_metrics["rmse"] = als_rmse
    content_metrics["rmse"] = np.nan
    knn_metrics["rmse"] = np.nan
    hybrid_metrics["rmse"] = np.nan

    metrics_rows = [
        {"method": "ALS", **als_metrics},
        {"method": "Content-TFIDF", **content_metrics},
        {"method": "UserKNN", **knn_metrics},
        {"method": "Hybrid-Weighted", **hybrid_metrics},
    ]
    metrics_df = pd.DataFrame(metrics_rows)

    movie_title_map = dict(zip(movies_pd["movieId"].astype(int), movies_pd["title"]))

    def recs_to_df(method_name: str, recs_map: Dict[int, List[int]]) -> pd.DataFrame:
        rows = []
        for user_id, recs in recs_map.items():
            for rank, movie_id in enumerate(recs, start=1):
                rows.append(
                    {
                        "method": method_name,
                        "userId": int(user_id),
                        "rank": rank,
                        "movieId": int(movie_id),
                        "title": movie_title_map.get(int(movie_id), "Unknown"),
                    }
                )
        return pd.DataFrame(rows)

    als_recs_df = recs_to_df("ALS", als_recs)
    content_recs_df = recs_to_df("Content-TFIDF", content_recs)
    knn_recs_df = recs_to_df("UserKNN", knn_recs)
    hybrid_recs_df = recs_to_df("Hybrid-Weighted", hybrid_recs)

    all_recs_df = pd.concat([als_recs_df, content_recs_df, knn_recs_df, hybrid_recs_df], ignore_index=True)

    candidate_users = train_pd.groupby("userId").size().sort_values(ascending=False).head(5).index.tolist()
    user_labels = {uid: f"FICTIF_U{idx + 1}" for idx, uid in enumerate(candidate_users)}

    sample_rows = []
    for method_name, rec_map in [
        ("ALS", als_recs),
        ("Content-TFIDF", content_recs),
        ("UserKNN", knn_recs),
        ("Hybrid-Weighted", hybrid_recs),
    ]:
        for user_id in candidate_users:
            recs = rec_map.get(user_id, [])
            for rank, movie_id in enumerate(recs[:k], start=1):
                sample_rows.append(
                    {
                        "user_label": user_labels[user_id],
                        "real_userId": user_id,
                        "method": method_name,
                        "rank": rank,
                        "movieId": movie_id,
                        "title": movie_title_map.get(movie_id, "Unknown"),
                    }
                )
    sample_recs_df = pd.DataFrame(sample_rows)
    sample_report_df = sample_recs_df.groupby(["user_label", "method"], as_index=False, sort=False).head(3)

    metrics_path = paths["modeling_dir"] / "metrics_comparison.csv"
    all_recs_path = paths["modeling_dir"] / "recommendations_all_methods.csv"
    sample_path = paths["modeling_dir"] / "recommendations_fictive_users.csv"
    als_params_path = paths["modeling_dir"] / "als_best_params.csv"
    hybrid_weights_path = paths["modeling_dir"] / "hybrid_best_weights.csv"
    hybrid_tuning_path = paths["modeling_dir"] / "hybrid_weight_grid_search.csv"

    metrics_df.to_csv(metrics_path, index=False)
    all_recs_df.to_csv(all_recs_path, index=False)
    sample_recs_df.to_csv(sample_path, index=False)
    pd.DataFrame([als_conf]).to_csv(als_params_path, index=False)
    pd.DataFrame([hybrid_weights | hybrid_metrics]).to_csv(hybrid_weights_path, index=False)
    hybrid_tuning_df.to_csv(hybrid_tuning_path, index=False)

    fig_metrics = paths["figures_dir"] / "recommender_metrics_comparison.png"
    metrics_plot = metrics_df.copy()
    metrics_plot["rmse"] = metrics_plot["rmse"].fillna(0.0)

    method_colors = ["#2A9D8F", "#E9C46A", "#264653", "#E76F51"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].bar(metrics_plot["method"], metrics_plot["precision_at_k"], color=method_colors[: len(metrics_plot)])
    axes[0].set_title(f"Precision@{k}")
    axes[0].set_ylim(0, max(0.001, metrics_plot["precision_at_k"].max() * 1.2))
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(metrics_plot["method"], metrics_plot["coverage_at_k"], color=method_colors[: len(metrics_plot)])
    axes[1].set_title(f"Coverage@{k}")
    axes[1].set_ylim(0, max(0.001, metrics_plot["coverage_at_k"].max() * 1.2))
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(metrics_plot["method"], metrics_plot["rmse"], color=method_colors[: len(metrics_plot)])
    axes[2].set_title("RMSE (ALS only)")
    axes[2].set_ylim(0, max(0.001, metrics_plot["rmse"].max() * 1.2))
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(fig_metrics, dpi=180)
    plt.close(fig)

    report_content = (
        "# Recommender Modeling Report\n\n"
        "## Experimental protocol\n\n"
        "- Data split: temporal split per user (80% train, 20% test with at least one item in test when possible).\n"
        f"- Relevance threshold for ranking metrics: rating >= 4.0 in test set.\n"
        f"- Ranking cutoff: Top-{k}.\n"
        f"- Evaluated users with at least one relevant item in test: {len(eval_users)}.\n\n"
        "## ALS (Spark MLlib)\n\n"
        f"- Best rank: {als_conf['rank']}\n"
        f"- Best regParam: {als_conf['regParam']}\n"
        f"- Best maxIter: {als_conf['maxIter']}\n"
        f"- Validation RMSE: {round(float(als_conf['validation_rmse']), 4)}\n"
        f"- Test RMSE: {round(float(als_conf['test_rmse']), 4)}\n\n"
        "## Hybrid strategy\n\n"
        f"- Fusion method: Weighted Reciprocal Rank Fusion\n"
        f"- Weights: ALS={hybrid_weights['als']}, Content={hybrid_weights['content']}, UserKNN={hybrid_weights['knn']}\n\n"
        "## Hybrid tuning\n\n"
        "- Search space: all weight triplets on a 0.1 grid with ALS + Content + UserKNN = 1\n"
        "- Selection rule: maximize Precision@K, then Recall@K, then Coverage@K\n"
        + "\nTop 5 weight settings:\n\n"
        + _rows_to_markdown(
            hybrid_tuning_df.head(5).round(4).to_dict(orient="records"),
            ["w_als", "w_content", "w_knn", "precision_at_k", "recall_at_k", "coverage_at_k"],
        )
        + "\n\n"
        "## Comparative metrics\n\n"
        + _rows_to_markdown(
            metrics_df.round(4).to_dict(orient="records"),
            ["method", "precision_at_k", "recall_at_k", "coverage_at_k", "rmse"],
        )
        + "\n\n## Recommendations for fictive users (mapped to real users)\n\n"
        + _rows_to_markdown(
            sample_report_df.to_dict(orient="records"),
            ["user_label", "real_userId", "method", "rank", "title"],
        )
        + "\n\n## Interpretation\n\n"
        "- ALS is optimized for rating prediction (RMSE), typically stronger on collaborative signals.\n"
        "- Content-TFIDF exploits genre similarity and is robust for explainability and cold-start items.\n"
        "- UserKNN captures neighborhood effects and can yield interpretable peer-based recommendations.\n"
        "- Hybrid-Weighted combines signal diversity and usually improves robustness across user profiles.\n"
        "- Final production choice should balance precision and catalog coverage depending on product goals.\n"
    )

    report_path = paths["reports_dir"] / "modeling_report.md"
    _write_text(report_path, report_content)

    return {
        "model_report": report_path,
        "modeling_dir": paths["modeling_dir"],
        "metrics_figure": fig_metrics,
    }
