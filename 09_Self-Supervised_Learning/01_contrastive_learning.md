# Contrastive Learning

Contrastive learning is a self-supervised approach that learns representations by comparing positive and negative pairs of samples. The goal is to bring representations of similar (positive) pairs closer and push those of dissimilar (negative) pairs apart in the embedding space.

## 1. Core Idea

Given a batch of data, we generate two augmented views for each sample. The model is trained to maximize the similarity between the representations of the two views of the same sample (positive pair) and minimize the similarity with other samples (negative pairs).

## 2. Mathematical Formulation

Let $`z_i`$ and $`z_j`$ be the representations of two augmented views of the same sample. The contrastive loss (NT-Xent) is:

```math
\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\mathrm{sim}(z_i, z_k)/\tau)}
```
where:
- $`\mathrm{sim}(z_i, z_j) = \frac{z_i^T z_j}{\|z_i\| \|z_j\|}`$ (cosine similarity)
- $`\tau`$ is a temperature parameter
- $`N`$ is the batch size

## 3. SimCLR

SimCLR is a simple framework for contrastive learning of visual representations. It uses strong data augmentations and a projection head to map representations to a space where contrastive loss is applied.

### SimCLR Pipeline
1. Apply two random augmentations to each image to create two correlated views.
2. Pass both views through an encoder (e.g., ResNet) to obtain representations.
3. Use a projection head (MLP) to map representations to a latent space.
4. Compute the NT-Xent loss.

### SimCLR Example (PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    similarity = torch.matmul(z, z.T)
    N = z.shape[0]
    mask = torch.eye(N, dtype=torch.bool).to(z.device)
    labels = torch.cat([
        torch.arange(z_i.size(0), 2 * z_i.size(0)),
        torch.arange(0, z_i.size(0))
    ]).to(z.device)
    similarity = similarity[~mask].view(N, -1)
    positives = torch.exp(similarity / temperature)
    denominator = torch.sum(torch.exp(similarity / temperature), dim=1)
    loss = -torch.log(positives / denominator)
    return loss.mean()
```

## 4. MoCo (Momentum Contrast)

MoCo maintains a dynamic dictionary (queue) of negative samples and uses a momentum encoder to update representations smoothly.

- **Key Encoder**: Updated with momentum from the query encoder.
- **Queue**: Stores a large set of negative keys for contrastive loss.

### MoCo Loss
The loss is similar to NT-Xent, but negatives are drawn from the queue.

### MoCo Example (PyTorch, simplified)
```python
class MoCo(nn.Module):
    def __init__(self, encoder, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = encoder
        self.K = K
        self.m = m
        self.T = T
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    # ... (forward and enqueue/dequeue logic)
```

## 5. CLIP (Contrastive Language-Image Pretraining)

CLIP learns joint representations of images and text by maximizing the similarity between matching image-text pairs and minimizing it for non-matching pairs.

### CLIP Loss
Given image features $`z_i`$ and text features $`t_i`$:

```math
\mathcal{L}_{\text{CLIP}} = -\frac{1}{N} \sum_{i=1}^N \left[ \log \frac{\exp(\mathrm{sim}(z_i, t_i)/\tau)}{\sum_{j=1}^N \exp(\mathrm{sim}(z_i, t_j)/\tau)} + \log \frac{\exp(\mathrm{sim}(t_i, z_i)/\tau)}{\sum_{j=1}^N \exp(\mathrm{sim}(t_i, z_j)/\tau)} \right]
```

### CLIP Example (PyTorch, simplified)
```python
import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    logits_per_image = image_features @ text_features.T / temperature
    logits_per_text = text_features @ image_features.T / temperature
    labels = torch.arange(image_features.size(0)).to(image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2
```

## 6. Summary

Contrastive learning is a powerful approach for learning representations from unlabeled data. By leveraging positive and negative pairs, models can learn features that generalize well to downstream tasks. 