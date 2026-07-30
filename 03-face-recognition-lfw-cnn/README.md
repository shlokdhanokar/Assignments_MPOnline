# Face Recognition using CNN — Labeled Faces in the Wild

A CNN trained from scratch on the LFW face dataset, benchmarked against the **classical eigenfaces pipeline (PCA + SVM)** that defined this task before deep learning.

**Result: the classical method wins decisively — eigenfaces 84.16% vs CNN 41.30%**, where the CNN barely clears the 41.15% majority-class floor. This is the most instructive outcome in the repository.

## Dataset

- **Source:** Labeled Faces in the Wild, University of Massachusetts
- **Link:** http://vis-www.cs.umass.edu/lfw/
- **Subset:** people with ≥ 70 images, resized to 40%
- **Size:** 1,288 images, 50×37 greyscale, 7 identities
- **Licence:** free for research use

Downloaded automatically via `sklearn.datasets.fetch_lfw_people` (~200 MB on first run). Not committed.

### The class imbalance is severe

| Person | Images | Share |
|---|---|---|
| **George W Bush** | **530** | **41.1%** |
| Colin Powell | 236 | 18.3% |
| Tony Blair | 144 | 11.2% |
| Donald Rumsfeld | 121 | 9.4% |
| Gerhard Schroeder | 109 | 8.5% |
| Ariel Sharon | 77 | 6.0% |
| Hugo Chavez | 71 | 5.5% |

Predicting "George W Bush" for every face scores **41.15%**. Every result below must be read against that floor — and one of them barely clears it.

## Libraries Used
tensorflow / keras · scikit-learn · numpy · matplotlib

## Methodology

1. Load the LFW subset; inspect class distribution and sample faces.
2. Scale pixels to 0–1; **stratified** 75/25 split (966 train / 322 test) so every identity appears in both halves. **Both models use identical train/test indices** — the comparison is like-for-like.
3. **Model A — CNN from scratch:** 3 conv blocks (32→64→128) with batch normalisation, progressive dropout, heavy augmentation (flip, ±6% rotation, zoom, translation), **global average pooling** instead of Flatten, class weighting, early stopping (patience 15) over 60 epochs.
4. **Model B — Eigenfaces:** `StandardScaler` → `PCA(150, whiten=True)` → `SVC(rbf, C=10, class_weight='balanced')`.
5. Evaluate both on accuracy, per-class recall, and a confusion matrix.

**Global average pooling is a deliberate choice** — a `Flatten` → `Dense` head would place most of the network's parameters in one layer and overfit 966 images almost immediately.

## Results

| Model | Test accuracy |
|---|---|
| **Eigenfaces (PCA + SVM)** | **0.8416** |
| CNN trained from scratch | 0.4130 |
| Majority-class floor | 0.4115 |

PCA retained **94.5% of pixel variance in 150 components**.

Plots: `sample_faces.png`, `eigenfaces.png` (the top 10 principal components rendered as face-like images), `training_history.png`, `confusion_matrix.png`, `sample_predictions.png`.

## Observations

1. **The CNN scored 0.4130 against a majority-class floor of 0.4115 — it learned essentially nothing.** A 0.15-point margin over "always guess George W Bush" means the network collapsed to the majority class despite class weighting and heavy augmentation. This is not a tuning miss; it is what a high-capacity model does when the data cannot constrain it.

2. **Eigenfaces more than doubles it at 0.8416**, and the reason is data volume, not cleverness. 966 training images across 7 identities is far below what a convolutional network needs to learn useful filters from scratch — CNNs earned their reputation on datasets of millions of images. PCA's linear compression into 150 components acts as a powerful prior, and the SVM's margin objective generalises well from few examples. Both degrade gracefully in exactly the regime where the CNN fails.

3. **"Deep learning is better" is a claim about a data regime, not a universal fact.** This is the most useful lesson in the repository: on ~1,000 small greyscale images, a 1990s technique beats a modern architecture by 43 points. The same CNN design works well on CIFAR-10 in project 02 — with 50,000 images.

4. **The imbalance shapes every metric.** With 41% of images being one person, accuracy alone is close to uninformative; the per-class recall table and confusion matrix are what expose the CNN's collapse, since a model that predicts one class perfectly and six classes never can still post a plausible-looking headline number.

## Conclusion

Two approaches to face recognition were compared on the LFW subset (1,288 images, 7 identities) using identical train/test splits: a CNN trained from scratch reached **41.30%**, and the classical eigenfaces pipeline of PCA plus an RBF SVM reached **84.16%**, against a **41.15%** majority-class floor.

The headline finding is that deep learning is not automatically the stronger choice. With 966 training images the CNN has far more capacity than the data can constrain, and it collapsed to predicting the majority identity despite augmentation, batch normalisation, global average pooling, class weighting and early stopping — all of which were necessary just to attempt the task. PCA's compression into 150 eigenfaces is a much stronger prior for this regime.

In production neither model would be trained from scratch. A face-embedding network pre-trained on millions of faces (FaceNet, ArcFace, VGGFace) reaches well above both numbers here, because the expensive part — learning what facial structure looks like — has already been paid for elsewhere, and only a small classifier needs fitting on these 966 images. That is the correct use of deep learning at this data scale: transfer, not training from scratch.

## How to Run

```bash
pip install -r requirements.txt
python face_recognition_lfw.py
```

Downloads LFW (~200 MB) on first run, then trains in a few minutes on CPU. Set `EPOCHS` to override.

## Files

| File | Purpose |
|---|---|
| `face_recognition_lfw.py` | Both models, identical splits, evaluation |
| `sample_faces.png` | Sample LFW images |
| `eigenfaces.png` | Top 10 principal components as face images |
| `training_history.png` | CNN accuracy/loss vs epoch, with the eigenfaces score as a reference line |
| `confusion_matrix.png` | Confusion matrix for the better model |
| `sample_predictions.png` | Correct and incorrect predictions |
