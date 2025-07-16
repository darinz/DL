# Pretext Tasks in Self-Supervised Learning

Pretext tasks are designed to generate supervised signals from unlabeled data, enabling neural networks to learn useful features. Here, we cover three common pretext tasks: Rotation Prediction, Jigsaw Puzzle, and Colorization.

> **Explanation:**
> Pretext tasks are clever tricks that turn unsupervised data into a supervised problem. By creating artificial labels (like rotation angle or patch order), we can train models to learn features that are useful for other tasks, even without real labels.

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

> **Math Breakdown:**
> - $`y_k`$: True label for class $`k`$ (1 if correct rotation, 0 otherwise).
> - $`\hat{y}_k`$: Model's predicted probability for class $`k`$.
> - $`K`$: Number of rotation classes (usually 4).
> - This is the cross-entropy loss, which encourages the model to assign high probability to the correct rotation.

### Step-by-Step Breakdown
1. **Rotate each image** by a random angle from the set {0°, 90°, 180°, 270°}.
   > **Explanation:**
   > Each image is randomly rotated, and the model must learn to recognize the correct orientation.
2. **Label** each rotated image with its rotation class.
   > **Explanation:**
   > The label is simply the index of the rotation (e.g., 0 for 0°, 1 for 90°, etc.).
3. **Train** the model to predict the rotation class from the image.
   > **Explanation:**
   > The model is trained using cross-entropy loss to predict the correct rotation.

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
> **Code Walkthrough:**
> - The function takes the model's predictions and the true rotation labels.
> - It computes the cross-entropy loss, which measures how well the model predicts the correct rotation.
> - This loss encourages the model to learn features that are sensitive to object orientation.

*This loss encourages the model to learn features sensitive to object orientation.*

> **Try it yourself!** Visualize the learned features. Do they capture object edges and shapes?

## 2. Jigsaw Puzzle

The model is trained to predict the correct arrangement of shuffled image patches.

> **Explanation:**
> By shuffling image patches and asking the model to predict the correct order, we force it to learn about spatial relationships and context in images.

> **Key Insight:** Solving jigsaw puzzles forces the model to learn spatial relationships and context.

### Mathematical Formulation
Let $`y`$ be the true permutation and $`\hat{y}`$ the predicted permutation probability:

```math
\mathcal{L}_{\text{jigsaw}} = - \sum_{k=1}^P y_k \log \hat{y}_k
```

> **Math Breakdown:**
> - $`y_k`$: True label for permutation $`k`$ (1 if correct, 0 otherwise).
> - $`\hat{y}_k`$: Model's predicted probability for permutation $`k`$.
> - $`P`$: Number of possible permutations (can be large for many patches).
> - This is the cross-entropy loss, encouraging the model to predict the correct arrangement.

### Step-by-Step Breakdown
1. **Divide** each image into patches (e.g., 3x3 grid).
   > **Explanation:**
   > The image is split into smaller pieces, like a puzzle.
2. **Shuffle** the patches according to a random permutation.
   > **Explanation:**
   > The patches are rearranged in a random order.
3. **Label** the shuffled image with its permutation index.
   > **Explanation:**
   > The label is the index of the permutation used to shuffle the patches.
4. **Train** the model to predict the permutation from the shuffled image.
   > **Explanation:**
   > The model is trained to recognize the correct arrangement using cross-entropy loss.

### Python Example (PyTorch)
```python
# Assume model outputs logits for each permutation
import torch.nn.functional as F

def jigsaw_loss(pred, target):
    return F.cross_entropy(pred, target)
```
> **Code Walkthrough:**
> - The function takes the model's predictions and the true permutation labels.
> - It computes the cross-entropy loss, which encourages the model to predict the correct arrangement of patches.

*This loss encourages the model to learn spatial and contextual cues.*

> **Common Pitfall:** Too many permutations can make the task too hard; too few can make it too easy.

## 3. Colorization

The model is trained to predict the color channels from a grayscale image.

> **Explanation:**
> Colorization tasks help the model learn about textures, semantics, and object boundaries by predicting color from grayscale.

> **Did you know?** Colorization pretext tasks help models learn texture, semantics, and object boundaries.

### Mathematical Formulation
Let $`x`$ be the original color image and $`\hat{x}`$ the predicted colorization:

```math
\mathcal{L}_{\text{color}} = \| x - \hat{x} \|^2
```

> **Math Breakdown:**
> - $`x`$: The true color image.
> - $`\hat{x}`$: The model's predicted color image.
> - The loss is the mean squared error between the true and predicted color images.

### Step-by-Step Breakdown
1. **Convert** the image to grayscale.
   > **Explanation:**
   > The input to the model is a grayscale version of the image.
2. **Train** the model to predict the original color channels from the grayscale input.
   > **Explanation:**
   > The model learns to add color back to the image, using context and semantics.
3. **Compute** the mean squared error between the predicted and true color images.
   > **Math Breakdown:**
   > The loss penalizes differences between the predicted and actual colors, encouraging accurate colorization.

### Python Example (PyTorch)
```python
import torch.nn as nn

def colorization_loss(pred, target):
    return ((pred - target) ** 2).mean()
```
> **Code Walkthrough:**
> - The function computes the mean squared error between the predicted and true color images.
> - This loss encourages the model to learn semantic and textural information from images.

*This loss encourages the model to learn semantic and textural information from images.*

> **Try it yourself!** Use the learned features from colorization as initialization for a downstream classification task.

## 4. Summary

Pretext tasks provide a way for neural networks to learn meaningful representations from unlabeled data, which can be transferred to downstream tasks. 

> **Key Insight:** The choice and design of pretext tasks can significantly impact the quality of learned representations. 