# Cancer Detection from Brain MRI Images

A convolutional neural network that classifies brain MRI scans into **glioma**, **meningioma**, **pituitary tumour**, or **no tumour** — evaluated the way a screening tool should be, on sensitivity rather than accuracy.

## Dataset

- **Source:** Brain Tumor Classification (MRI), Sartaj Bhuvaji et al.
- **Kaggle:** https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
- **Public mirror:** https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet
- **Size:** 2,870 training + 394 test images, 4 classes
- **Not committed** to this repository.

Training distribution: glioma 826 (28.8%), meningioma 822 (28.6%), pituitary 827 (28.8%), **no_tumor 395 (13.8%)**. The healthy class is the smallest, which matters — a model biased toward predicting *some* tumour would look accurate while being useless at ruling disease out.

### Getting the data

```bash
# From the public mirror (no Kaggle account needed)
curl -L -o mri.zip https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/archive/refs/heads/master.zip
unzip mri.zip     # creates Brain-Tumor-Classification-DataSet-master/
```

## Libraries Used
tensorflow / keras · numpy · scikit-learn · matplotlib

## Methodology

1. Load `Training/` and `Testing/` with `image_dataset_from_directory` at 150×150 RGB. The dataset's own `Testing/` folder is the held-out set and is touched exactly once.
2. Hold 15% of `Training/` out for validation.
3. Rescale pixels **inside the model**, so the identical transform applies at inference and cannot be forgotten.
4. Augment mildly — horizontal flip, ±5% rotation, 10% zoom, 10% contrast.
5. Train a 4-block CNN (32→64→128→128) with batch normalisation, progressive dropout, and global average pooling.
6. Weight classes inversely to frequency to counteract the small healthy class.
7. Evaluate with accuracy, per-class recall, a confusion matrix, and a **binary tumour/no-tumour clinical view**.

### Two decisions worth explaining

**Augmentation is deliberately mild.** Aggressive flips and rotations are wrong for MRI — anatomical orientation is diagnostic information, not noise. A vertically flipped brain is not a plausible training example.

**BatchNorm momentum is 0.9, not the Keras default of 0.99 — this was a real bug.** BatchNorm keeps running mean/variance for inference. At momentum 0.99 those statistics need roughly 500 steps to track the batch statistics, but this dataset provides only ~77 steps per epoch. The result was that training and validation were effectively using *different normalisation*:

| BatchNorm momentum | Train accuracy | Validation accuracy |
|---|---|---|
| 0.99 (Keras default) | 0.7893 | **0.2279** ← chance for 4 classes |
| **0.9 (fixed)** | 0.8943 | **0.8837** |

Validation accuracy sitting exactly at chance while training accuracy climbed past 0.78 is the signature of this failure — it is not ordinary overfitting, which would degrade gradually. Without the fix this project would have shipped a model that looked trained and predicted at random on anything unseen.

## Results

| Metric | Value |
|---|---|
| Validation accuracy (held-out 15% of Training) | **0.8837** |
| **Test accuracy** (dataset's own `Testing/` folder) | **0.6218** |
| Test images | 394 |

### Clinical view — tumour present vs absent

| | Count |
|---|---|
| True positives (tumour found) | 196 |
| **False negatives (tumour missed)** | **93** |
| True negatives (correctly clear) | 100 |
| False positives (false alarm) | 5 |
| **Sensitivity** (recall on tumour) | **0.6782** |
| **Specificity** (recall on healthy) | **0.9524** |

### Per-class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| glioma | 1.0000 | **0.1000** | 0.1818 | 100 |
| meningioma | 0.6567 | 0.7652 | 0.7068 | 115 |
| no_tumor | 0.5181 | 0.9524 | 0.6711 | 105 |
| pituitary | 0.8246 | 0.6351 | 0.7176 | 74 |

Plots: `sample_scans.png`, `training_history.png`, `confusion_matrix.png`, `sample_predictions.png`.

## Observations

1. **Validation says 88%, the test folder says 62% — and that 27-point gap is the headline result.** Validation is a random split of `Training/`, so it shares that folder's distribution. The dataset's separate `Testing/` folder does not. A model selected on validation alone would have been reported as far better than it is.

2. **Glioma recall collapses to 10% with precision 1.0.** The model almost never predicts glioma on the test set, but is always right when it does. That asymmetry points at distribution shift rather than an inability to learn the class — glioma is learned well enough during training but the test folder's glioma images do not resemble it. This dataset has **documented labelling problems in the glioma folder**, which is the most likely cause.

3. **Sensitivity (67.8%) is far below specificity (95.2%), which is backwards for screening.** The model misses 93 of 289 genuine tumours while raising only 5 false alarms. Clinically these errors are not interchangeable: a false alarm costs a follow-up scan, a miss can cost a life. A deployed tool would move its decision threshold hard toward sensitivity, accepting many more false alarms to reduce misses.

4. **Accuracy is the wrong headline for a medical model,** and this run demonstrates why: 62% accuracy conceals both a near-perfect healthy-class recall and a near-total glioma failure. The per-class breakdown and the binary clinical view are what actually describe the model's behaviour.

## Conclusion

A CNN trained from scratch reached **88.4% validation accuracy** but only **62.2%** on the dataset's own held-out test folder, with **67.8% sensitivity** and **95.2% specificity** on the underlying screening question. Class weighting was necessary because the healthy class is the smallest, and fixing the BatchNorm momentum was necessary for the model to function at all.

The two results that matter are negative ones. The validation-to-test gap shows that a random split of the training folder is not a substitute for a genuinely held-out set. The glioma collapse — 10% recall at 100% precision — is consistent with the labelling problems documented in this dataset rather than a modelling failure.

These numbers should be read as an architecture and evaluation exercise, not a clinical validation. Every sample is a single 2D slice rather than a volumetric study; a real diagnostic tool would use 3D volumes, multiple MRI sequences, transfer learning from a medical imaging backbone, a threshold tuned for sensitivity, and would assist a radiologist rather than replace one.

## How to Run

```bash
pip install -r requirements.txt
# place Brain-Tumor-Classification-DataSet-master/ in this folder (see above)
python brain_tumor_mri.py
```

Roughly 25–30 minutes on CPU for 25 epochs (~70 s/epoch). Set `EPOCHS` to override.

## Files

| File | Purpose |
|---|---|
| `brain_tumor_mri.py` | Full pipeline — load, augment, train, evaluate |
| `sample_scans.png` | Sample MRI images by class |
| `training_history.png` | Accuracy and loss vs epoch |
| `confusion_matrix.png` | 4-class confusion matrix |
| `sample_predictions.png` | Correct and incorrect predictions |
