# Contrastive Learning

Contrastive learning is a self-supervised approach that learns representations by comparing positive and negative pairs of samples. The goal is to bring representations of similar (positive) pairs closer and push those of dissimilar (negative) pairs apart in the embedding space.

> **Explanation:**
> Contrastive learning is a way for neural networks to learn useful features without labels. By comparing pairs of data, the model learns what makes things similar or different, which helps it generalize to new tasks.

> **Key Insight:** Contrastive learning leverages the structure of data itself, without labels, to learn powerful and generalizable features.

> **Did you know?** Many state-of-the-art models in vision and language use contrastive learning as a pretraining step!

## 1. Core Idea

Given a batch of data, we generate two augmented views for each sample. The model is trained to maximize the similarity between the representations of the two views of the same sample (positive pair) and minimize the similarity with other samples (negative pairs).

> **Geometric Intuition:** Imagine each data point as a dot in space. Contrastive learning pulls together dots that are "views" of the same item and pushes apart dots that are different items.

## 2. Mathematical Formulation

Let $`z_i`$ and $`z_j`$ be the representations of two augmented views of the same sample. The contrastive loss (NT-Xent) is:

```math
\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\mathrm{sim}(z_i, z_k)/\tau)}
```

> **Math Breakdown:**
> - $`\mathrm{sim}(z_i, z_j) = \frac{z_i^T z_j}{\|z_i\| \|z_j\|}`$: Cosine similarity measures how aligned two vectors are, regardless of their magnitude.
> - $`\tau`$: Temperature parameter that controls how sharply similarities are weighted.
> - $`N`$: Batch size; $`2N`$ because each sample has two views.
> - The numerator is the similarity between the positive pair; the denominator sums over all possible pairs except the anchor itself.
> - The loss encourages positive pairs to be close and negative pairs to be far apart in the embedding space.

### Step-by-Step Breakdown
1. **Augment each sample** to create two views: $`x_i^{(1)}`$, $`x_i^{(2)}`$.
   > **Explanation:**
   > Data augmentations (e.g., cropping, color jitter) create different but related versions of the same input.
2. **Encode** both views: $`z_i = f(x_i^{(1)})`$, $`z_j = f(x_i^{(2)})`$.
   > **Explanation:**
   > The encoder network maps each view to a feature vector (embedding).
3. **Compute similarities** between all pairs in the batch.
   > **Math Breakdown:**
   > For each anchor, compare its embedding to all others using cosine similarity.
4. **Apply the NT-Xent loss** to maximize similarity for positive pairs and minimize for negatives.
   > **Explanation:**
   > The loss function ensures that only the correct pairs are pulled together, while all others are pushed apart.

> **Common Pitfall:** If augmentations are too weak, the model may not learn invariance. If too strong, positive pairs may become too different.

## 3. SimCLR

SimCLR is a simple framework for contrastive learning of visual representations. It uses strong data augmentations and a projection head to map representations to a space where contrastive loss is applied.

### SimCLR Pipeline
1. Apply two random augmentations to each image to create two correlated views.
2. Pass both views through an encoder (e.g., ResNet) to obtain representations.
3. Use a projection head (MLP) to map representations to a latent space.
4. Compute the NT-Xent loss.

> **Try it yourself!** Experiment with different augmentations (color jitter, crop, blur) and see how they affect learned representations.

### SimCLR Example (PyTorch)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.5):
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)  # Normalize to unit sphere
    similarity = torch.matmul(z, z.T)  # Cosine similarity matrix
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
> **Code Walkthrough:**
> - Concatenate the two sets of embeddings (for both views) into one batch.
> - Normalize embeddings so that cosine similarity is just a dot product.
> - Compute the similarity matrix for all pairs.
> - Remove self-similarities (the diagonal of the matrix).
> - For each anchor, the positive is the corresponding augmented view; all others are negatives.
> - The loss is computed to maximize the similarity for positives and minimize for negatives.

*This function computes the NT-Xent loss for a batch of positive and negative pairs.*

## 4. MoCo (Momentum Contrast)

MoCo maintains a dynamic dictionary (queue) of negative samples and uses a momentum encoder to update representations smoothly.

- **Key Encoder**: Updated with momentum from the query encoder.
- **Queue**: Stores a large set of negative keys for contrastive loss.

> **Explanation:**
> MoCo's queue allows the model to use many more negative samples than would fit in a single batch, which improves the quality of learned representations. The momentum encoder ensures that the representations in the queue change smoothly over time, making training more stable.

> **Key Insight:** The queue allows MoCo to use many more negative samples than fit in a single batch, improving representation quality.

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
> **Code Walkthrough:**
> - The model has two encoders: a query encoder (updated by gradient descent) and a key encoder (updated by momentum).
> - The queue stores a large number of negative samples for contrastive loss.
> - The momentum update ensures the key encoder changes slowly, stabilizing the queue.
> - The rest of the logic (not shown) handles adding new keys and removing old ones from the queue.

*MoCo's momentum encoder and queue enable scalable and stable contrastive learning.*

> **Try it yourself!** Vary the queue size $`K`$ and momentum $`m`$. How do these affect performance and stability?

## 5. CLIP (Contrastive Language-Image Pretraining)

CLIP learns joint representations of images and text by maximizing the similarity between matching image-text pairs and minimizing it for non-matching pairs.

> **Explanation:**
> CLIP is trained on pairs of images and their corresponding text descriptions. The model learns to align the two modalities in a shared embedding space, so that matching pairs are close together and mismatched pairs are far apart.

### CLIP Loss
Given image features $`z_i`$ and text features $`t_i`$:

```math
\mathcal{L}_{\text{CLIP}} = -\frac{1}{N} \sum_{i=1}^N \left[ \log \frac{\exp(\mathrm{sim}(z_i, t_i)/\tau)}{\sum_{j=1}^N \exp(\mathrm{sim}(z_i, t_j)/\tau)} + \log \frac{\exp(\mathrm{sim}(t_i, z_i)/\tau)}{\sum_{j=1}^N \exp(\mathrm{sim}(t_i, z_j)/\tau)} \right]
```

> **Math Breakdown:**
> - For each image-text pair, the numerator is the similarity between the matching pair.
> - The denominator sums over all possible pairs, so the loss encourages only the correct pairs to be close.
> - The loss is symmetric: it is computed both from the image's and the text's perspective.
> - $`\tau`$ is a temperature parameter that controls the sharpness of the similarity distribution.

### Step-by-Step Breakdown
1. **Encode images and texts** to get $`z_i`$ and $`t_i`$.
   > **Explanation:**
   > Use separate encoders for images and text to get their embeddings.
2. **Compute similarities** between all image-text pairs.
   > **Math Breakdown:**
   > Calculate cosine similarity between every image and every text in the batch.
3. **Apply the CLIP loss** to maximize similarity for matching pairs and minimize for mismatches.
   > **Explanation:**
   > The loss function ensures that only the correct image-text pairs are close in the embedding space.

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
> **Code Walkthrough:**
> - Normalize image and text features so that cosine similarity is just a dot product.
> - Compute logits for all image-text and text-image pairs.
> - Use cross-entropy loss to encourage correct pairs to have the highest similarity.
> - The final loss is the average of the image-to-text and text-to-image losses.

*CLIP's loss encourages alignment between image and text representations in a shared embedding space.*

> **Did you know?** CLIP can perform zero-shot classification: just provide a text description, and it can find matching images!

## 6. Summary

Contrastive learning is a powerful approach for learning representations from unlabeled data. By leveraging positive and negative pairs, models can learn features that generalize well to downstream tasks. 

> **Key Insight:** The choice of augmentations, batch size, and negative sampling strategy are crucial for effective contrastive learning. 