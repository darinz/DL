# Self-Supervised Learning in Neural Networks

Self-supervised learning (SSL) is a paradigm in machine learning where the model learns useful representations from unlabeled data by solving pretext tasks. These tasks are designed such that the supervision signal is generated from the data itself, enabling the model to learn without explicit human-annotated labels. SSL has become foundational in modern neural network research, especially in computer vision and natural language processing.

## 1. Contrastive Learning

Contrastive learning aims to learn representations by bringing similar (positive) pairs closer and pushing dissimilar (negative) pairs apart in the embedding space. The core idea is to maximize agreement between differently augmented views of the same data sample.

### Key Methods
- **SimCLR**: Uses data augmentations and a contrastive loss to learn representations.
- **MoCo**: Maintains a dynamic dictionary as a queue of data samples to enable large and consistent sets of negative samples.
- **CLIP**: Learns joint representations of images and text using contrastive loss.

### Contrastive Loss (NT-Xent)
The normalized temperature-scaled cross entropy loss (NT-Xent) is commonly used:

```math
\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\mathrm{sim}(z_i, z_k)/\tau)}
```
where $`\mathrm{sim}(z_i, z_j)`$ is the cosine similarity between representations $`z_i`$ and $`z_j`$, and $`\tau`$ is a temperature parameter.

## 2. Masked Autoencoding

Masked autoencoding methods train neural networks to reconstruct missing or masked parts of the input data. This encourages the model to learn global and local structures in the data.

### Key Methods
- **MAE (Masked Autoencoders)**: Randomly masks patches of the input (e.g., image) and trains the model to reconstruct the missing content.
- **SimMIM**: Similar to MAE, but with different masking and reconstruction strategies.

### Objective
Given an input $`x`$ and a mask $`M`$, the model predicts the masked content:

```math
\mathcal{L}_{\text{MAE}} = \mathbb{E}_{x, M} \left[ \| f(x \odot (1 - M)) - x \odot M \|^2 \right]
```
where $`\odot`$ denotes element-wise multiplication.

## 3. Pretext Tasks

Pretext tasks are designed to create supervised signals from unlabeled data. The model is trained to solve these tasks, which encourages learning of useful features.

### Common Pretext Tasks
- **Rotation Prediction**: Predict the rotation angle applied to an image (e.g., 0°, 90°, 180°, 270°).
- **Jigsaw Puzzle**: Predict the correct arrangement of shuffled image patches.
- **Colorization**: Predict the color channels from grayscale images.

### Example: Rotation Prediction Loss
Let $`y`$ be the true rotation label and $`\hat{y}`$ the predicted label:

```math
\mathcal{L}_{\text{rot}} = - \sum_{k=1}^K y_k \log \hat{y}_k
```
where $`K`$ is the number of rotation classes.

---

Self-supervised learning enables neural networks to leverage vast amounts of unlabeled data, leading to improved performance and generalization in downstream tasks. 