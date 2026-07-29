# Movie Recommendation System

Four recommendation strategies built and compared on MovieLens — popularity baselines, item-item collaborative filtering, matrix factorisation via SVD, and content-based filtering on genres — evaluated on both **rating prediction** and **top-N ranking**.

## Dataset

- **Source:** GroupLens — MovieLens Latest Small
- **Link:** https://grouplens.org/datasets/movielens/
- **Direct download:** https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- **Size:** 100,836 ratings · 610 users · 9,724 rated films (9,742 catalogue)
- **Licence:** free for research and education (see the dataset's `README.txt`)

Downloaded automatically on first run. Not committed.

## Libraries Used
pandas · numpy · scikit-learn · matplotlib

## The core problem: sparsity

| Property | Value |
|---|---|
| Ratings | 100,836 |
| Users | 610 |
| Films rated | 9,724 |
| Matrix density | **1.70 %** |
| Matrix emptiness | **98.30 %** |
| Mean rating | 3.5016 |

Every method below is really a strategy for filling in the 98.3 % of the user–item matrix that is missing.

## Methodology

1. **Load and explore** ratings and film metadata; measure sparsity and the rating distribution.
2. **Split** 80/20, **stratified by user**, so every user appears in both splits. An unstratified split would leave some users entirely unseen at training time, and the comparison would then measure cold-start handling instead of recommendation quality.
3. **Build four families of model:**
   - *Baselines* — global mean, per-user mean, per-item mean shrunk toward the global mean.
   - *Item-item CF* — cosine similarity over **user-mean-centred** rating vectors.
   - *Matrix factorisation* — truncated SVD, 50 latent factors, on the centred matrix.
   - *Content-based* — TF-IDF over genre tokens, cosine similarity between films.
4. **Evaluate twice:** RMSE/MAE for rating prediction, and Precision@10 / Recall@10 / NDCG@10 for ranking.

### Two implementation details that decide whether this works

**Centre by user mean before computing similarity.** Without it, similarity is dominated by *how generously a user rates* rather than by what they actually enjoyed — two users who rate everything 5 look identical regardless of taste.

**Shrink the item-CF denominator.** The item-CF prediction is `user_mean + Σ(sim × deviation) / Σ|sim|`. For a film with almost no similarity mass to anything the user has rated, that denominator is near zero, the ratio explodes on numerical noise, and obscure titles top every recommendation list. Adding a constant (`CF_SHRINKAGE = 2.0`) pulls weakly-supported items back toward the user's own mean in proportion to how little evidence supports them. Before this fix, item-CF scored **0.0005** Precision@10 — effectively random. After it, **0.0470**.

A minimum-support rule (`≥ 5 training ratings`, leaving 3,232 of 9,724 films recommendable) is applied **identically to every model**, so the comparison stays fair.

## Results

### Rating prediction

| Model | RMSE | MAE |
|---|---|---|
| **Item-item CF** | **0.8603** | **0.6542** |
| SVD (50 factors) | 0.9222 | 0.7160 |
| User mean | 0.9413 | 0.7347 |
| Item mean (shrunk) | 0.9643 | 0.7512 |
| Global mean | 1.0386 | 0.8260 |

### Top-10 ranking (562 users scored)

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|
| **Popularity** | **0.1294** | 0.0985 | **0.1722** |
| SVD (50 factors) | 0.1274 | **0.1139** | 0.1716 |
| Item-item CF | 0.0470 | 0.0380 | 0.0591 |

Plots: `model_comparison.png`, `ranking_metrics.png`.

### Sample output

For user 1 (186 films rated; top-rated include *Se7en*, *The Usual Suspects*, *Rob Roy*), the SVD model recommends:

```
 1. Sixth Sense, The (1999)                      predicted 4.80
 2. Lord of the Rings: The Two Towers (2002)     predicted 4.72
 3. Lord of the Rings: Return of the King (2003) predicted 4.69
 4. Lord of the Rings: Fellowship of the Ring    predicted 4.67
 5. Godfather: Part II, The (1974)               predicted 4.66
```

Content-based, films most similar to *Toy Story (1995)* by genre: *Antz*, *Toy Story 2*, *Shrek the Third*, *The Good Dinosaur*.

## Observations

1. **Sparsity sets the terms.** Predicting the global mean for every pair already gives 1.0386 RMSE. The useful question is how much of the *remaining* error each method removes, not whether it beats random.
2. **Item-item CF wins rating prediction** (0.8603 RMSE) — a 17 % improvement over the global mean and clearly ahead of SVD's 0.9222.
3. **RMSE and ranking disagree, and that is the most important result here.** Item-item CF has the *best* RMSE and the *worst* ranking (0.0591 NDCG vs 0.1716 for SVD). This is a well-documented property of neighbourhood methods: they predict a known rating accurately, but score most unrated items close to the user's mean, giving them little to discriminate between when forced to produce a ranked list. Users are shown a list, not a predicted number — so a model can win the metric everyone reports and still recommend badly.
4. **The popularity baseline is deceptively strong**, essentially tying SVD on NDCG. Blockbusters are both widely rated and widely liked, so "recommend what's popular" is a genuinely hard bar. Any personalised system must be measured against it — one that merely reproduces it adds no personalisation value while starving the long tail.

## Conclusion

Four recommendation strategies were compared on MovieLens (100,836 ratings, 610 users, 9,724 films). Item-item collaborative filtering minimised rating error at **0.8603 RMSE**, while popularity and SVD tied for the best top-10 ranking at roughly **0.172 NDCG**. The headline lesson is that the two evaluation modes rank the models differently, and only one of them reflects what users experience.

Collaborative filtering learns from behaviour and can surface a film sharing no genre with anything the user has watched, but it cannot score a title nobody has rated — the cold-start problem. That is exactly the gap content-based filtering fills, since TF-IDF over genres needs no ratings at all. A production system blends all three: content-based coverage for new items, collaborative signal for anything with enough history, and popularity as the fallback when neither applies.

## How to Run

```bash
pip install -r requirements.txt
python movie_recommender.py
```

Runs in about 1–2 minutes. Note it materialises a 9,724 × 9,724 similarity matrix, so roughly 1 GB of free RAM is needed.

## Files

| File | Purpose |
|---|---|
| `movie_recommender.py` | All four models, both evaluations, sample recommendations |
| `model_comparison.png` | Rating distribution and RMSE comparison |
| `ranking_metrics.png` | Precision / Recall / NDCG @10 |
| `requirements.txt` | Dependencies |
