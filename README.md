# AI/ML Assignments and Projects — MPOnline

Nine machine-learning projects spanning classical ML, deep learning, computer vision, reinforcement learning, recommender systems, deployment, and a Retrieval-Augmented Generation capstone.

Every project is **self-contained** — its own script, README, dependencies, and dataset link — and **tested end to end**, with the reported numbers produced by actually running the code rather than quoted from a tutorial.

## Projects

| # | Project | Domain | Headline result |
|---|---|---|---|
| 1 | [Adult Census Income Classification](01-adult-census-income-classification/) | Tabular classification | 87.23% accuracy, 0.9269 ROC-AUC |
| 2 | [CIFAR-10 Image Classification (CNN)](02-cifar10-image-classification-cnn/) | Computer vision | CNN vs dense-network control |
| 3 | [Face Recognition (LFW, CNN)](03-face-recognition-lfw-cnn/) | Computer vision | CNN vs eigenfaces baseline |
| 4 | [Cancer Detection from MRI](04-cancer-detection-mri/) | Medical imaging | 4-class tumour classification |
| 5 | [Cart-Pole RL Agent](05-cartpole-rl-agent/) | Reinforcement learning | DQN from scratch |
| 6 | [Lunar Lander RL Agent](06-lunar-lander-rl-agent/) | Reinforcement learning | Double DQN |
| 7 | [Movie Recommendation System](07-movie-recommendation-system/) | Recommender systems | 0.8603 RMSE, 4 methods compared |
| 8 | [End-to-End Render Deployment](08-end-to-end-render-deployment/) | MLOps | Flask + JSON API, 14/14 tests pass |
| 9 | [RAG Chatbot (Capstone)](09-rag-chatbot-capstone/) | LLM / IR | 95% Hit@1, 0.967 MRR |

## Datasets

No dataset is committed to this repository. Each project either downloads its data automatically on first run or links to the source in its README.

| # | Dataset | Source | Acquisition |
|---|---|---|---|
| 1 | Adult Census Income | [UCI](https://archive.ics.uci.edu/dataset/2/adult) | Automatic |
| 2 | CIFAR-10 | [Krizhevsky / CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) | Automatic (Keras) |
| 3 | Labeled Faces in the Wild | [LFW / UMass](http://vis-www.cs.umass.edu/lfw/) | Automatic (scikit-learn) |
| 4 | Brain Tumor MRI | [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) | Manual / public mirror |
| 5–6 | Gymnasium environments | [Gymnasium](https://gymnasium.farama.org/) | Simulated |
| 7 | MovieLens (100k) | [GroupLens](https://grouplens.org/datasets/movielens/) | Automatic |
| 8 | Adult Census Income | [UCI](https://archive.ics.uci.edu/dataset/2/adult) | Automatic (build step) |
| 9 | Authored ML corpus | Ships in `knowledge_base/` | Included |

## Running a project

Each folder is independent:

```bash
cd 01-adult-census-income-classification
pip install -r requirements.txt
python adult_census_income.py
```

Every script prints a structured walkthrough — data understanding, preprocessing, model development, evaluation, observations, and conclusion — and writes its plots as PNGs alongside the code.

## Environment

Developed and tested on **Python 3.13, CPU only** (no GPU). Core stack: scikit-learn, TensorFlow/Keras, PyTorch, Gymnasium, pandas, matplotlib, Flask.

Training times reflect CPU execution — the deep-learning projects use modest architectures and early stopping so they complete in minutes rather than hours.

## Notes on method

A few conventions applied consistently across the nine projects, because they change whether the reported numbers mean anything:

- **Validation data is never the test set.** Where a training curve is plotted, the validation split comes out of the *training* data, so the reported test score is untouched by model selection.
- **Preprocessing is fitted on training data only**, inside a `Pipeline` where the library supports it, so no test information leaks through scaling or imputation statistics.
- **Baselines are included.** A metric without a floor is not informative: the imbalanced classifier is compared against the majority-class rate, the CNN against a dense network of similar size, the recommender against a popularity baseline, and the RL agents against a random policy.
- **Negative and awkward results are reported.** Where a classical method beats a neural network, where a model wins on one metric and loses on another, or where a training run collapses, the README says so and explains why — those are usually the most informative parts of a project.

## Repository layout

```
Assignments_MPOnline/
├── 01-adult-census-income-classification/
├── 02-cifar10-image-classification-cnn/
├── 03-face-recognition-lfw-cnn/
├── 04-cancer-detection-mri/
├── 05-cartpole-rl-agent/
├── 06-lunar-lander-rl-agent/
├── 07-movie-recommendation-system/
├── 08-end-to-end-render-deployment/
├── 09-rag-chatbot-capstone/
└── README.md
```
