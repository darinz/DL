# Pretext Tasks in Self-Supervised Learning

Pretext tasks are designed to generate supervised signals from unlabeled data, enabling neural networks to learn useful features. Here, we cover three common pretext tasks: Rotation Prediction, Jigsaw Puzzle, and Colorization.

> **Key Insight:** Pretext tasks transform unsupervised data into a supervised learning problem, allowing models to learn transferable features without human labels.

> **Did you know?** Many breakthroughs in self-supervised learning started with creative pretext tasks like predicting image rotations or solving jigsaw puzzles!

## 1. Rotation Prediction

The model is trained to predict the rotation angle applied to an image. Typical angles are 0°, 90°, 180°, and 270°.

> **Geometric Intuition:** By forcing the model to recognize rotated objects, it learns to understand object shapes and orientations.

### Mathematical Formulation
Let $`y`$ be the true rotation label (one-hot) and $`\hat{y}`$ the predicted label:

```math
\mathcal{L}_{\text{rot}} = - \sum_{k=1}^K y_k \log \hat{y}_k
```
where $`K`$ is the number of rotation classes.

### Step-by-Step Breakdown
1. **Rotate each image** by a random angle from the set {0°, 90°, 180°, 270°}.
2. **Label** each rotated image with its rotation class.
3. **Train** the model to predict the rotation class from the image.

### Python Example (PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotation_loss(pred, target):
    return F.cross_entropy(pred, target)

# Example usage:
# pred = model(rotated_images)
# loss = rotation_loss(pred, rotation_labels)
```
*This loss encourages the model to learn features sensitive to object orientation.*

> **Try it yourself!** Visualize the learned features. Do they capture object edges and shapes?

## 2. Jigsaw Puzzle

The model is trained to predict the correct arrangement of shuffled image patches.

> **Key Insight:** Solving jigsaw puzzles forces the model to learn spatial relationships and context.

### Mathematical Formulation
Let $`y`$ be the true permutation and $`\hat{y}`$ the predicted permutation probability:

```math
\mathcal{L}_{\text{jigsaw}} = - \sum_{k=1}^P y_k \log \hat{y}_k
```
where $`P`$ is the number of possible permutations.

### Step-by-Step Breakdown
1. **Divide** each image into patches (e.g., 3x3 grid).
2. **Shuffle** the patches according to a random permutation.
3. **Label** the shuffled image with its permutation index.
4. **Train** the model to predict the permutation from the shuffled image.

### Python Example (PyTorch)
```python
# Assume model outputs logits for each permutation
import torch.nn.functional as F

def jigsaw_loss(pred, target):
    return F.cross_entropy(pred, target)
```
*This loss encourages the model to learn spatial and contextual cues.*

> **Common Pitfall:** Too many permutations can make the task too hard; too few can make it too easy.

## 3. Colorization

The model is trained to predict the color channels from a grayscale image.

> **Did you know?** Colorization pretext tasks help models learn texture, semantics, and object boundaries.

### Mathematical Formulation
Let $`x`$ be the original color image and $`\hat{x}`$ the predicted colorization:

```math
\mathcal{L}_{\text{color}} = \| x - \hat{x} \|^2
```

### Step-by-Step Breakdown
1. **Convert** the image to grayscale.
2. **Train** the model to predict the original color channels from the grayscale input.
3. **Compute** the mean squared error between the predicted and true color images.

### Python Example (PyTorch)
```python
import torch.nn as nn

def colorization_loss(pred, target):
    return ((pred - target) ** 2).mean()
```
*This loss encourages the model to learn semantic and textural information from images.*

> **Try it yourself!** Use the learned features from colorization as initialization for a downstream classification task.

## 4. Summary

Pretext tasks provide a way for neural networks to learn meaningful representations from unlabeled data, which can be transferred to downstream tasks. 

> **Key Insight:** The choice and design of pretext tasks can significantly impact the quality of learned representations. 