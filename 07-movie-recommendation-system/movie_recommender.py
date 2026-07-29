"""Movie Recommendation System on MovieLens 100K (ml-latest-small).

Builds and compares four approaches:
  1. Popularity / mean baselines
  2. Item-item collaborative filtering (cosine similarity)
  3. Matrix factorisation (truncated SVD on the mean-centred rating matrix)
  4. Content-based filtering (TF-IDF over genres)

Rating prediction is scored with RMSE/MAE; top-N ranking is scored with
Precision@K, Recall@K and NDCG@K, because a recommender is judged on the
ordering it produces, not only on the ratings it predicts.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = "ml-latest-small"
TEST_SIZE = 0.2
RANDOM_STATE = 42
SVD_COMPONENTS = 50
TOP_K = 10
RELEVANT_THRESHOLD = 4.0     # a held-out rating >= 4 counts as a genuine "hit"
MIN_TEST_RATINGS = 5         # users need enough held-out items for ranking metrics
MIN_ITEM_SUPPORT = 5         # a film needs this many training ratings to be recommendable
CF_SHRINKAGE = 2.0           # similarity mass added to the item-CF denominator

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#d8d7d2"


def style_axes(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_SECONDARY, length=0)


def ensure_dataset(project_dir: Path) -> Path:
    data_path = project_dir / DATA_DIR
    if data_path.exists():
        return data_path
    print(f"Downloading MovieLens from {DATA_URL} ...")
    try:
        payload = urllib.request.urlopen(DATA_URL, timeout=180).read()
        zipfile.ZipFile(io.BytesIO(payload)).extractall(project_dir)
    except Exception as error:  # noqa: BLE001
        raise SystemExit(f"Could not download MovieLens: {error}") from error
    return data_path


def precision_recall_ndcg_at_k(recommended: list[int], relevant: set[int], k: int):
    """Ranking metrics for one user's top-k list."""
    top_k = recommended[:k]
    hits = [1.0 if item in relevant else 0.0 for item in top_k]
    precision = sum(hits) / k
    recall = sum(hits) / len(relevant) if relevant else 0.0

    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return precision, recall, ndcg


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = ensure_dataset(project_dir)

    # ---------------------------------------------------------------- 1. Load
    print("=" * 72)
    print("1. DATA UNDERSTANDING")
    print("=" * 72)
    ratings = pd.read_csv(data_path / "ratings.csv")
    movies = pd.read_csv(data_path / "movies.csv")

    print("Ratings - first 5 records:")
    print(ratings.head())
    print()
    print("Movies - first 5 records:")
    print(movies.head())
    print()

    n_users = ratings["userId"].nunique()
    n_movies_rated = ratings["movieId"].nunique()
    density = len(ratings) / (n_users * n_movies_rated)
    print(f"Ratings      : {len(ratings):,}")
    print(f"Users        : {n_users:,}")
    print(f"Movies rated : {n_movies_rated:,}  (catalogue: {len(movies):,})")
    print(f"Rating scale : {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"Mean rating  : {ratings['rating'].mean():.4f}")
    print(f"Matrix density: {density:.2%}  (i.e. {1 - density:.2%} of the "
          f"user-item matrix is empty)")
    print()
    print("Sparsity is the defining property of this problem: every method below is")
    print("really a strategy for filling in 98%+ missing entries.")
    print()

    print("Ratings per user  - median %.0f, min %d, max %d"
          % (ratings.groupby("userId").size().median(),
             ratings.groupby("userId").size().min(),
             ratings.groupby("userId").size().max()))
    print("Ratings per movie - median %.0f, min %d, max %d"
          % (ratings.groupby("movieId").size().median(),
             ratings.groupby("movieId").size().min(),
             ratings.groupby("movieId").size().max()))
    print()

    popular = (ratings.groupby("movieId")
               .agg(count=("rating", "size"), mean=("rating", "mean"))
               .merge(movies[["movieId", "title"]], on="movieId")
               .sort_values("count", ascending=False))
    print("Top 5 most-rated movies:")
    print(popular.head(5)[["title", "count", "mean"]].to_string(index=False))
    print()

    # ------------------------------------------------------- 2. Preprocessing
    print("=" * 72)
    print("2. PREPROCESSING AND SPLIT")
    print("=" * 72)
    print("Missing values:", int(ratings.isnull().sum().sum()))

    train, test = train_test_split(
        ratings, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=ratings["userId"],
    )
    print(f"Train ratings: {len(train):,}   Test ratings: {len(test):,}")
    print("Stratified by user, so every user appears in both splits - otherwise")
    print("collaborative filtering would face cold-start users at test time and the")
    print("comparison would measure cold-start handling instead of recommendation quality.")
    print()

    user_ids = np.sort(ratings["userId"].unique())
    movie_ids = np.sort(ratings["movieId"].unique())
    user_index = {u: i for i, u in enumerate(user_ids)}
    movie_index = {m: i for i, m in enumerate(movie_ids)}
    title_of = movies.set_index("movieId")["title"].to_dict()

    matrix = np.zeros((len(user_ids), len(movie_ids)), dtype=np.float32)
    for user, movie, rating in zip(train["userId"], train["movieId"], train["rating"]):
        matrix[user_index[user], movie_index[movie]] = rating
    observed = matrix > 0
    print(f"Training user-item matrix: {matrix.shape} "
          f"({observed.sum():,} observed cells)")
    print()

    global_mean = float(train["rating"].mean())
    user_means = np.array([
        matrix[i][observed[i]].mean() if observed[i].any() else global_mean
        for i in range(len(user_ids))
    ], dtype=np.float32)
    item_counts = observed.sum(axis=0)
    item_sums = matrix.sum(axis=0)
    # Shrink sparsely-rated items toward the global mean, otherwise a single
    # 5-star rating makes an obscure film look like the best in the catalogue.
    shrinkage = 10.0
    item_means = (item_sums + shrinkage * global_mean) / (item_counts + shrinkage)

    test_users = test["userId"].map(user_index).to_numpy()
    test_items = test["movieId"].map(movie_index).to_numpy()
    test_truth = test["rating"].to_numpy()

    # ------------------------------------------------------------- 3. Models
    print("=" * 72)
    print("3. MODEL DEVELOPMENT")
    print("=" * 72)
    predictions: dict[str, np.ndarray] = {}

    # (a) Baselines
    predictions["Global mean"] = np.full(len(test_truth), global_mean, dtype=np.float32)
    predictions["User mean"] = user_means[test_users]
    predictions["Item mean (shrunk)"] = item_means[test_items]
    print("(a) Baselines: global mean, per-user mean, shrunk per-item mean")

    # (b) Item-item collaborative filtering
    # Centre each user's ratings so 'generous rater' bias does not dominate similarity.
    centred = np.where(observed, matrix - user_means[:, None], 0.0).astype(np.float32)
    item_similarity = cosine_similarity(centred.T).astype(np.float32)
    np.fill_diagonal(item_similarity, 0.0)

    print(f"(b) Item-item CF: {item_similarity.shape} cosine similarity matrix")

    def predict_item_cf(u: int, i: int) -> float:
        rated = np.flatnonzero(observed[u])
        if rated.size == 0:
            return global_mean
        sims = item_similarity[i, rated]
        keep = sims > 0
        if not keep.any():
            return float(user_means[u])
        sims = sims[keep]
        deviations = centred[u, rated[keep]]
        return float(user_means[u] + np.dot(sims, deviations) / sims.sum())

    predictions["Item-item CF"] = np.array(
        [predict_item_cf(u, i) for u, i in zip(test_users, test_items)], dtype=np.float32
    )

    # (c) Matrix factorisation via truncated SVD on the centred matrix
    u_mat, sigma, vt = np.linalg.svd(centred, full_matrices=False)
    u_k = u_mat[:, :SVD_COMPONENTS]
    s_k = np.diag(sigma[:SVD_COMPONENTS])
    vt_k = vt[:SVD_COMPONENTS, :]
    reconstructed = (u_k @ s_k @ vt_k) + user_means[:, None]
    explained = (sigma[:SVD_COMPONENTS] ** 2).sum() / (sigma ** 2).sum()
    print(f"(c) SVD matrix factorisation: {SVD_COMPONENTS} latent factors, "
          f"{explained:.1%} of the centred matrix's energy retained")
    predictions[f"SVD ({SVD_COMPONENTS} factors)"] = reconstructed[test_users, test_items]

    # (d) Content-based filtering on genres
    movies_in_matrix = movies[movies["movieId"].isin(movie_ids)].copy()
    movies_in_matrix["genres"] = movies_in_matrix["genres"].str.replace("|", " ", regex=False)
    ordered = movies_in_matrix.set_index("movieId").reindex(movie_ids)
    genre_text = ordered["genres"].fillna("").to_numpy()
    tfidf = TfidfVectorizer(token_pattern=r"[^\s]+")
    genre_features = tfidf.fit_transform(genre_text)
    content_similarity = cosine_similarity(genre_features)
    np.fill_diagonal(content_similarity, 0.0)
    print(f"(d) Content-based: TF-IDF over {len(tfidf.vocabulary_)} genre tokens")
    print()

    for name, values in predictions.items():
        predictions[name] = np.clip(values, 0.5, 5.0)

    # ---------------------------------------------------------- 4. Evaluation
    print("=" * 72)
    print("4. EVALUATION - RATING PREDICTION")
    print("=" * 72)
    rating_scores = {}
    for name, values in predictions.items():
        rating_scores[name] = {
            "RMSE": float(np.sqrt(mean_squared_error(test_truth, values))),
            "MAE": float(mean_absolute_error(test_truth, values)),
        }
    rating_table = pd.DataFrame(rating_scores).T.sort_values("RMSE")
    print(rating_table.round(4).to_string())
    print()
    best_rating_model = rating_table.index[0]
    print(f"Best by RMSE: {best_rating_model}")
    print()

    # Top-N ranking evaluation
    print("=" * 72)
    print(f"4b. EVALUATION - TOP-{TOP_K} RANKING")
    print("=" * 72)
    test_by_user: dict[int, list[tuple[int, float]]] = {}
    for u, i, r in zip(test_users, test_items, test_truth):
        test_by_user.setdefault(int(u), []).append((int(i), float(r)))

    # Applied identically to every model, so the comparison stays fair: a film with
    # fewer than 5 training ratings is not a recommendation, it is a guess.
    eligible = item_counts >= MIN_ITEM_SUPPORT
    print(f"Recommendable catalogue: {int(eligible.sum()):,} of {len(movie_ids):,} films "
          f"have at least {MIN_ITEM_SUPPORT} training ratings")
    print()

    def rank_scores(score_matrix: np.ndarray) -> dict[str, float]:
        precisions, recalls, ndcgs = [], [], []
        for u, entries in test_by_user.items():
            if len(entries) < MIN_TEST_RATINGS:
                continue
            relevant = {i for i, r in entries if r >= RELEVANT_THRESHOLD}
            if not relevant:
                continue
            scores = score_matrix[u].copy()
            scores[observed[u]] = -np.inf   # never recommend an already-rated film
            scores[~eligible] = -np.inf     # nor one with too little training support
            recommended = list(np.argsort(-scores)[:TOP_K])
            p, r, n = precision_recall_ndcg_at_k(recommended, relevant, TOP_K)
            precisions.append(p); recalls.append(r); ndcgs.append(n)
        return {
            f"Precision@{TOP_K}": float(np.mean(precisions)),
            f"Recall@{TOP_K}": float(np.mean(recalls)),
            f"NDCG@{TOP_K}": float(np.mean(ndcgs)),
            "Users scored": len(precisions),
        }

    popularity_scores = np.tile(item_means * np.log1p(item_counts), (len(user_ids), 1))
    # Item-based CF scores for every (user, item) pair. The denominator must sum
    # similarity over the items THIS user actually rated - normalising by the
    # column total instead would divide by similarity mass the user never touched.
    cf_numerator = centred @ item_similarity
    cf_denominator = observed.astype(np.float32) @ np.abs(item_similarity)
    # Shrinkage. A film with little similarity mass to anything the user rated has a
    # tiny denominator, so the ratio explodes on numerical noise and obscure titles top
    # the chart. Adding a constant pulls weakly-supported items back toward the user's
    # own mean, in proportion to how little evidence supports them.
    cf_denominator = cf_denominator + CF_SHRINKAGE
    item_cf_scores = np.clip(user_means[:, None] + cf_numerator / cf_denominator, 0.5, 5.0)

    ranking_models = {
        "Popularity": popularity_scores,
        "Item-item CF": item_cf_scores,
        f"SVD ({SVD_COMPONENTS} factors)": reconstructed,
    }
    ranking_table = pd.DataFrame({name: rank_scores(m) for name, m in ranking_models.items()}).T
    print(ranking_table.round(4).to_string())
    print()
    print("Ranking is the metric that matters operationally: users see a list, not a")
    print("predicted score. A model can win on RMSE and still rank badly.")
    print()

    # -------------------------------------------------- 5. Sample recommendations
    print("=" * 72)
    print("5. SAMPLE RECOMMENDATIONS")
    print("=" * 72)
    # A typical user, not the heaviest rater: someone who has rated 2,000 films has
    # almost nothing left to recommend, which makes for a meaningless demo.
    ratings_per_user = observed.sum(axis=1)
    typical = np.flatnonzero((ratings_per_user >= 80) & (ratings_per_user <= 200))
    demo_user = int(typical[0]) if typical.size else int(np.argmax(ratings_per_user))
    liked = [(movie_ids[i], matrix[demo_user, i]) for i in np.flatnonzero(observed[demo_user])]
    liked.sort(key=lambda pair: -pair[1])
    print(f"User {user_ids[demo_user]} rated {len(liked)} films. Top rated:")
    for movie_id, rating in liked[:5]:
        print(f"   {rating:.1f}  {title_of.get(movie_id, movie_id)}")
    print()

    svd_scores = reconstructed[demo_user].copy()
    svd_scores[observed[demo_user]] = -np.inf
    svd_scores[~eligible] = -np.inf
    print(f"Top {TOP_K} SVD recommendations for this user:")
    for rank, i in enumerate(np.argsort(-svd_scores)[:TOP_K], start=1):
        print(f"  {rank:2d}. {title_of.get(movie_ids[i], movie_ids[i]):<55s} "
              f"predicted {svd_scores[i]:.2f}")
    print()

    seed_title = "Toy Story (1995)"
    seed_ids = [m for m, t in title_of.items() if t == seed_title and m in movie_index]
    if seed_ids:
        seed = movie_index[seed_ids[0]]
        print(f"Content-based - films most similar to '{seed_title}' by genre:")
        for rank, i in enumerate(np.argsort(-content_similarity[seed])[:5], start=1):
            print(f"  {rank}. {title_of.get(movie_ids[i], movie_ids[i])}")
        print()

    # ------------------------------------------------------------- Plots
    fig, (ax_dist, ax_rmse) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    counts = ratings["rating"].value_counts().sort_index()
    ax_dist.bar(counts.index.astype(str), counts.values, color=COLORS[0], width=0.7)
    ax_dist.set_title("Rating distribution", color=INK_PRIMARY)
    ax_dist.set_xlabel("Rating", color=INK_SECONDARY)
    ax_dist.set_ylabel("Count", color=INK_SECONDARY)
    style_axes(ax_dist)

    order = rating_table.sort_values("RMSE", ascending=False)
    ax_rmse.barh(order.index, order["RMSE"], color=COLORS[0])
    for y, value in enumerate(order["RMSE"]):
        ax_rmse.text(value + 0.01, y, f"{value:.4f}", va="center",
                     color=INK_SECONDARY, fontsize=9)
    ax_rmse.set_title("Rating prediction error (lower is better)", color=INK_PRIMARY)
    ax_rmse.set_xlabel("RMSE", color=INK_SECONDARY)
    ax_rmse.set_xlim(0, order["RMSE"].max() * 1.18)
    style_axes(ax_rmse)
    fig.tight_layout()
    fig.savefig(project_dir / "model_comparison.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: model_comparison.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = [f"Precision@{TOP_K}", f"Recall@{TOP_K}", f"NDCG@{TOP_K}"]
    x = np.arange(len(metrics))
    width = 0.26
    for offset, (name, color) in enumerate(zip(ranking_table.index, COLORS)):
        ax.bar(x + (offset - 1) * width, ranking_table.loc[name, metrics].to_numpy(),
               width, label=name, color=color)
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title(f"Top-{TOP_K} ranking quality", color=INK_PRIMARY, fontsize=12)
    ax.set_ylabel("Score", color=INK_SECONDARY)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(project_dir / "ranking_metrics.png", dpi=150, facecolor="white")
    plt.close(fig)
    print("Saved: ranking_metrics.png")
    print()

    # -------------------------------------------------------- Observations
    best_rank_model = ranking_table[f"NDCG@{TOP_K}"].idxmax()
    svd_rmse = rating_table.loc[f"SVD ({SVD_COMPONENTS} factors)", "RMSE"]
    item_rmse = rating_table.loc["Item-item CF", "RMSE"]
    global_rmse = rating_table.loc["Global mean", "RMSE"]

    print("=" * 72)
    print("OBSERVATIONS")
    print("=" * 72)
    print(
        f"1. The user-item matrix is {1 - density:.1%} empty, and that single fact shapes "
        f"everything. Predicting the global mean for every pair already gives "
        f"{global_rmse:.4f} RMSE, so the useful question is how much of the remaining "
        f"error each method removes - not whether it beats random."
    )
    print(
        f"2. {best_rating_model} wins on rating prediction: item-item CF reaches "
        f"{item_rmse:.4f} RMSE and SVD {svd_rmse:.4f}, against {global_rmse:.4f} for the "
        f"global mean. Centring by user mean before computing "
        "similarity matters here: without it, similarity is dominated by whether a user "
        "rates generously rather than by what they actually enjoyed."
    )
    print(
        f"3. RMSE and ranking disagree, which is the most important lesson in this "
        f"project. {best_rank_model} gives the best NDCG@{TOP_K} "
        f"({ranking_table.loc[best_rank_model, f'NDCG@{TOP_K}']:.4f}) even though the "
        "RMSE ordering is different. Users are shown a ranked list, so a model that "
        "predicts every rating slightly better can still surface worse recommendations."
    )
    print(
        "4. The popularity baseline is deceptively strong on ranking, because blockbusters "
        "are both widely rated and widely liked. Any recommender must be judged against it - "
        "beating popularity is the real bar, and a system that merely reproduces it adds "
        "no personalisation value while quietly starving the long tail."
    )
    print()
    print("=" * 72)
    print("CONCLUSION")
    print("=" * 72)
    print(
        f"Four recommendation strategies were built and compared on MovieLens "
        f"({len(ratings):,} ratings, {n_users} users, {n_movies_rated:,} films). "
        f"{best_rating_model} minimised rating error at {rating_table.loc[best_rating_model, 'RMSE']:.4f} "
        f"RMSE, while {best_rank_model} produced the best top-{TOP_K} ranking at NDCG "
        f"{ranking_table.loc[best_rank_model, f'NDCG@{TOP_K}']:.4f}. Collaborative filtering "
        "learns from behaviour and can surface a film sharing no genre with anything the "
        "user has seen, but it cannot rate a title nobody has rated - the cold-start "
        "problem - which is exactly the gap content-based filtering fills, since TF-IDF "
        "over genres needs no ratings at all. A production system therefore blends them: "
        "content-based coverage for new items, collaborative signal for everything with "
        "enough history, and popularity as the fallback when neither applies."
    )


if __name__ == "__main__":
    main()
