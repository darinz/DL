# Pretext Tasks in Self-Supervised Learning

Pretext tasks are designed to generate supervised signals from unlabeled data, enabling neural networks to learn useful features. Here, we cover three common pretext tasks: Rotation Prediction, Jigsaw Puzzle, and Colorization.

## 1. Rotation Prediction

The model is trained to predict the rotation angle applied to an image. Typical angles are 0°, 90°, 180°, and 270°.

### Mathematical Formulation
Let $`y`$ be the true rotation label (one-hot) and $`\hat{y}`$ the predicted label:

```math
\mathcal{L}_{\text{rot}} = - \sum_{k=1}^K y_k \log \hat{y}_k
```
where $`K`$ is the number of rotation classes.

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

## 2. Jigsaw Puzzle

The model is trained to predict the correct arrangement of shuffled image patches.

### Mathematical Formulation
Let $`y`$ be the true permutation and $`\hat{y}`$ the predicted permutation probability:

```math
\mathcal{L}_{\text{jigsaw}} = - \sum_{k=1}^P y_k \log \hat{y}_k
```
where $`P`$ is the number of possible permutations.

### Python Example (PyTorch)
```python
# Assume model outputs logits for each permutation
import torch.nn.functional as F

def jigsaw_loss(pred, target):
    return F.cross_entropy(pred, target)
```

## 3. Colorization

The model is trained to predict the color channels from a grayscale image.

### Mathematical Formulation
Let $`x`$ be the original color image and $`\hat{x}`$ the predicted colorization:

```math
\mathcal{L}_{\text{color}} = \| x - \hat{x} \|^2
```

### Python Example (PyTorch)
```python
import torch.nn as nn

def colorization_loss(pred, target):
    return ((pred - target) ** 2).mean()
```

## 4. Summary

Pretext tasks provide a way for neural networks to learn meaningful representations from unlabeled data, which can be transferred to downstream tasks. 