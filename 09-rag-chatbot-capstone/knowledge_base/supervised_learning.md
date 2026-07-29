# Supervised Learning

Supervised learning trains a model on labelled examples, where each input is paired with the correct output. The model learns a mapping from inputs to outputs and is then used to predict labels for unseen data.

## Classification vs Regression

Classification predicts a discrete category — spam or not spam, which of ten digits an image shows. Regression predicts a continuous number — a house price, a temperature.

## Common algorithms

Logistic regression fits a linear decision boundary and outputs calibrated probabilities. It is fast, interpretable, and a strong baseline, but cannot capture interactions between features unless they are engineered by hand.

Decision trees split the feature space by asking a sequence of yes/no questions. They capture non-linear relationships and interactions automatically, but a single deep tree has high variance and overfits easily.

Random forests train many decision trees on bootstrapped samples with random feature subsets, then average their votes. Averaging de-correlated trees reduces variance substantially.

Gradient boosting builds trees sequentially, where each new tree corrects the errors of the ensemble so far. It usually outperforms random forests on tabular data but is more sensitive to hyperparameters.

Support Vector Machines find the hyperplane that maximises the margin between classes. The kernel trick lets them fit non-linear boundaries by implicitly mapping data into a higher-dimensional space.

K-Nearest Neighbours classifies a point by majority vote among its k closest training examples. It requires no training but prediction is slow and it degrades badly in high dimensions.

## Evaluation

Accuracy is the fraction of correct predictions. On imbalanced data it is misleading — if 99 percent of samples are negative, predicting negative always scores 99 percent.

Precision is the fraction of positive predictions that are correct. Recall is the fraction of actual positives that were found. F1-score is their harmonic mean. ROC-AUC measures ranking quality independent of any decision threshold.

A confusion matrix shows counts of true positives, false positives, true negatives and false negatives, exposing exactly which classes are being confused.

Class imbalance means one class vastly outnumbers another. Remedies include class weighting, resampling, and choosing a decision threshold from the relative cost of each error type rather than leaving it at 0.5.
