# Adult Census Income Classification

Predict whether a person earns more than **$50,000/year** from 1994 US census attributes — a classic imbalanced binary classification problem.

## Dataset

- **Source:** UCI Machine Learning Repository — Adult Data Set
- **Link:** https://archive.ics.uci.edu/dataset/2/adult
- **Direct files:** https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
- **Size:** 32,561 training + 16,281 test records, 14 attributes
- **Licence:** public domain (UCI), free to use

The script **downloads the dataset automatically** on first run, so no manual setup is needed. The raw files are not committed to this repository.

## Libraries Used
pandas · numpy · scikit-learn · matplotlib

## Methodology

1. **Load** the canonical `adult.data` / `adult.test` split. The test file needs two fixes most tutorials miss: it carries a junk first line, and its labels have a trailing period (`<=50K.`), so a naive load silently produces four classes instead of two.
2. **Explore** — shapes, dtypes, summary statistics, class balance.
3. **Preprocess:**
   - `?` is parsed as missing (1,836 missing `workclass`, 1,843 `occupation`, 583 `native-country`).
   - Drop `fnlwgt` — a census *sampling weight* describing how many people the row represents. It is an artefact of survey design, not a personal attribute, and has no business being a predictor.
   - Numerical: median imputation → `StandardScaler`.
   - Categorical: most-frequent imputation → one-hot encoding with `min_frequency=10` to stop rare `native-country` values becoming near-unique identifiers.
   - All preprocessing lives in a `Pipeline`, so it is fitted on training data only and cannot leak test information.
4. **Train** three classifiers: Logistic Regression, Random Forest (300 trees), Gradient Boosting (`HistGradientBoostingClassifier`, 300 iterations).
5. **Evaluate** with accuracy, precision, recall, F1 and ROC-AUC; plot ROC curves, the confusion matrix and permutation importance.

**Why the canonical split rather than a random one:** the dataset ships a fixed train/test split, and using it makes these numbers directly comparable with published benchmarks (~87 % is the well-established ceiling for non-tuned models here).

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8512 | 0.7299 | 0.5874 | 0.6509 | 0.9036 |
| Random Forest | 0.8619 | 0.7579 | 0.6105 | 0.6763 | 0.9153 |
| **Gradient Boosting** | **0.8723** | **0.7733** | **0.6500** | **0.7063** | **0.9269** |

Best model: **Gradient Boosting**.

```
Classification report - Gradient Boosting
              precision    recall  f1-score   support
       <=50K     0.8968    0.9411    0.9184     12435
        >50K     0.7733    0.6500    0.7063      3846
    accuracy                         0.8723     16281

Confusion matrix
[[11702   733]     <=50K correctly / wrongly flagged as >50K
 [ 1346  2500]]    >50K missed / correctly caught
```

**Top features (permutation importance, drop in ROC-AUC when shuffled):**

| Feature | Importance |
|---|---|
| marital-status | 0.0874 |
| capital-gain | 0.0572 |
| age | 0.0540 |
| education-num | 0.0307 |
| capital-loss | 0.0141 |

Plots: `roc_curves.png`, `confusion_matrix.png`, `feature_importance.png`.

## Observations

1. **Accuracy is a misleading headline here.** Predicting `<=50K` for everybody already scores **76.4 %**, because only 23.6 % of the test set earns more. Gradient Boosting's 87.2 % is a real gain over that baseline, but ROC-AUC (0.9269) and minority-class recall are the metrics that actually distinguish the three models.
2. **Recall on the minority class is the weak point.** The best model catches only 2,500 of 3,846 genuine high earners (65 % recall) while raising 733 false alarms. That is what a 0.5 probability threshold buys — it optimises accuracy, not balanced error cost.
3. **Tree ensembles beat the linear model** by ~2.3 points of AUC, because income depends on *interactions* (marital status combined with hours and education) that logistic regression can only capture if they are hand-engineered.
4. **`capital-gain` is a skewed flag, not a smooth predictor.** Most people record zero, so it effectively marks a small wealthy subgroup rather than varying continuously across the population — worth knowing before treating its importance score as "capital gains drive income."

## Conclusion

Three classifiers were trained on the 1994 Adult Census extract using its canonical 32,561/16,281 split, with Gradient Boosting best at **87.23 % accuracy** and **0.9269 ROC-AUC**. The defining property of this dataset is class imbalance: a do-nothing classifier already scores 76.4 %, so headline accuracy means little on its own. Marital status, capital gains, age and education carry the most signal.

Two things matter before any real deployment: the decision threshold should be set from the relative cost of false positives versus false negatives rather than left at 0.5, and the model should be audited for disparate impact across the `sex` and `race` columns — this data encodes 1994 social patterns, and a model trained on it will reproduce them.

## How to Run

```bash
pip install -r requirements.txt
python adult_census_income.py
```

Runs in roughly 1–2 minutes on CPU (permutation importance is the slowest step). The dataset downloads automatically on first run.

## Files

| File | Purpose |
|---|---|
| `adult_census_income.py` | Full pipeline: load → preprocess → train → evaluate |
| `roc_curves.png` | ROC curves for all three models |
| `confusion_matrix.png` | Confusion matrix for the best model |
| `feature_importance.png` | Permutation importance |
| `requirements.txt` | Dependencies |
