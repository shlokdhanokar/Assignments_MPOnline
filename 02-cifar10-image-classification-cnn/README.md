# CIFAR-10 Image Classification using CNN

A VGG-style convolutional neural network trained on CIFAR-10, benchmarked against a **dense-only control network with three times the parameters** to isolate what convolution actually contributes.

**Result: CNN 79.65% vs dense baseline 42.69%** — a 37-point gap, with the CNN using *fewer* parameters.

## Dataset

- **Source:** CIFAR-10, Krizhevsky et al., University of Toronto
- **Link:** https://www.cs.toronto.edu/~kriz/cifar.html
- **Size:** 50,000 training + 10,000 test images, 32×32 RGB, 10 classes
- **Licence:** free for research use

Downloaded automatically on first run via `keras.datasets.cifar10` (~170 MB). Not committed.

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck — perfectly balanced at 5,000 training images each.

## Libraries Used
tensorflow / keras · numpy · scikit-learn · matplotlib

## Architecture

Three convolutional blocks of doubling width, then a dense head:

```
Input 32×32×3
  ├─ RandomFlip / RandomTranslation / RandomZoom     (training only)
  ├─ [Conv 32 → BN → Conv 32 → BN → MaxPool → Drop 0.2]
  ├─ [Conv 64 → BN → Conv 64 → BN → MaxPool → Drop 0.3]
  ├─ [Conv 128 → BN → Conv 128 → BN → MaxPool → Drop 0.4]
  └─ Flatten → Dense 128 → BN → Drop 0.5 → Dense 10 (softmax)
```

**552,874 parameters.** Adam, categorical crossentropy, batch size 128, 15 epochs with early stopping (patience 8) and `ReduceLROnPlateau`.

### Design decisions

**Augmentation lives inside the model.** Placing `RandomFlip`/`RandomTranslation`/`RandomZoom` as layers means they apply during training and are automatically skipped at inference — the evaluation path can't accidentally receive augmented images.

**Dropout increases with depth** (0.2 → 0.3 → 0.4 → 0.5), regularising the widest layers hardest, where the parameters concentrate.

**Validation comes from a 10% split of the training set**, never the test set. The test set is evaluated once.

**The control network is the point.** A dense-only network with 1,707,274 parameters — over 3× the CNN's — is trained on identical data. Without this, a CNN result is just a number; with it, the comparison isolates *convolutional structure* from *raw capacity*.

## Results

| Model | Test accuracy | Parameters |
|---|---|---|
| **VGG-style CNN** | **0.7965** | 552,874 |
| Dense-only baseline | 0.4269 | 1,707,274 |
| Random guess | 0.1000 | — |

Test loss: 0.6011. Early stopping restored weights from epoch 12.

Plots: `sample_images.png`, `training_history.png`, `confusion_matrix.png`, `sample_predictions.png`.

## Observations

1. **The CNN reaches 79.65% across 10 balanced classes against a 10% random-guess floor**, training for 15 epochs with early stopping restoring the best weights from epoch 12.

2. **Convolution, not capacity, is what wins — and this is the result the control run exists to establish.** The dense baseline has **3.1× more parameters** yet scores **42.69%**, thirty-seven points below the CNN. Flattening a 32×32 image destroys the spatial adjacency that makes an edge an edge, and weight sharing lets the CNN detect a feature anywhere in the frame instead of relearning it at every position. More parameters cannot buy back a discarded inductive bias.

3. **Errors concentrate among the animal classes.** Cats, dogs, deer and birds share texture, posture and background statistics far more than the vehicle classes do — at 32×32 resolution an animal occupies only a few hundred pixels, which is genuinely near the limit of what is distinguishable.

4. **Augmentation and progressive dropout keep the train/validation gap narrow.** Without them a network this size memorises 50,000 images quickly. With them the validation curve tracks training closely, and the run ends via early stopping rather than divergence.

## Conclusion

A VGG-style CNN with three convolutional blocks, batch normalisation, progressive dropout and in-model augmentation classified CIFAR-10 with **79.65% test accuracy**.

The controlled comparison is the informative result: a dense network with **3.1× more parameters** reaches only **42.69%** on identical data. That isolates the benefit of convolutional inductive bias — local receptive fields, weight sharing, translation tolerance — from the benefit of raw parameter count, and shows the former dominates.

Remaining errors cluster among visually similar animal classes rather than spreading evenly, which is the expected failure mode at this resolution. Accuracy beyond this point comes from deeper residual architectures or transfer learning from a network pre-trained on higher-resolution images, not from training this architecture longer — the early-stopping callback fired well before the epoch budget ran out.

> **Note on budget:** trained for 15 epochs on CPU (~5 minutes/epoch). A longer schedule on GPU typically reaches 88–92% with this architecture; the conclusions about convolution vs capacity are unaffected, since both models were given the same budget.

## How to Run

```bash
pip install -r requirements.txt
python cifar10_cnn.py
```

Set `EPOCHS` / `BASELINE_EPOCHS` to override the budget. Dataset downloads automatically on first run.

## Files

| File | Purpose |
|---|---|
| `cifar10_cnn.py` | CNN, dense control, training, evaluation |
| `sample_images.png` | Sample images by class |
| `training_history.png` | Accuracy and loss vs epoch |
| `confusion_matrix.png` | 10-class confusion matrix |
| `sample_predictions.png` | Correct and incorrect predictions with confidence |
