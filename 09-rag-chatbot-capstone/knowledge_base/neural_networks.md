# Neural Networks and Deep Learning

An artificial neural network is composed of layers of units. Each unit computes a weighted sum of its inputs, adds a bias, and applies a non-linear activation function.

## Why hidden layers matter

Hidden layers compose the previous layer's outputs into progressively more abstract features. In image recognition the first layer may detect edges, the second strokes and corners, and the third whole shapes. Without a hidden layer and a non-linear activation, a network collapses to a linear model no matter how many layers it has.

## Activation functions

ReLU outputs the input if positive and zero otherwise. It is the default for hidden layers because it does not saturate for positive values, which keeps gradients flowing. Softmax converts a vector of scores into a probability distribution over classes and is used in the output layer for multi-class classification. Sigmoid squashes values into zero to one and is used for binary classification.

## Training

Backpropagation computes the gradient of the loss with respect to every weight by applying the chain rule backwards through the network. An optimiser such as Adam or SGD then updates the weights in the direction that reduces the loss.

Categorical crossentropy is the standard loss for multi-class classification. Binary crossentropy is used for two-class problems, and mean squared error for regression.

Normalising pixel values to the range zero to one before training matters because large input magnitudes produce correspondingly large weighted sums and gradients, so the optimiser overshoots and converges erratically.

## Convolutional Neural Networks

A CNN applies small learned filters across an image. Two properties make it far more effective than a dense network on images: local receptive fields, which respect the fact that nearby pixels are related, and weight sharing, which lets a feature be detected anywhere in the frame instead of being relearned at every position. Pooling layers downsample the feature maps, giving tolerance to small translations.

## Regularisation

Dropout randomly zeroes a fraction of units during training, forcing the network not to rely on any single feature. Batch normalisation standardises the inputs of each layer, which stabilises and speeds up training. Data augmentation applies random flips, rotations and crops to training images, effectively enlarging the dataset. Early stopping halts training when validation performance stops improving.

## Overfitting

Overfitting happens when a model memorises the training data instead of learning generalisable patterns. The signature is training loss continuing to fall while validation loss rises.

## Deep learning versus traditional machine learning

The central advantage of deep learning is that features are learned from the data rather than hand-engineered. Its main limitations are that it needs large labelled datasets and considerable compute, and that a trained network is a black box offering no human-readable justification for any single prediction. On small datasets a classical method with a strong prior, such as PCA followed by an SVM, can outperform a network trained from scratch.
