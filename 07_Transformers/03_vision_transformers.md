# Vision Transformers (ViT)

Vision Transformers (ViT) adapt the Transformer architecture for image data, enabling state-of-the-art performance on computer vision tasks.

## 1. Overview

ViT splits an image into fixed-size patches, flattens them, and projects them into a sequence of embeddings, which are then processed by a standard Transformer encoder.

## 2. Patch Embedding
Given an image $`x \in \mathbb{R}^{H \times W \times C}`$, it is split into $`N`$ patches of size $`P \times P`$:

```math
N = \frac{HW}{P^2}
```

Each patch is flattened and projected to a $`D`$-dimensional embedding.

### Python Example: Patch Embedding
```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x

# Example usage
img = torch.randn(1, 3, 224, 224)
patch_embed = PatchEmbedding(224, 16, 3, 768)
patches = patch_embed(img)
print(patches.shape)  # (1, 196, 768)
```

## 3. Position Embedding

As with NLP Transformers, ViT adds position embeddings to the patch embeddings to retain spatial information.

## 4. Transformer Encoder

The sequence of patch embeddings is processed by a standard Transformer encoder, as described in the Transformer Architecture guide.

## 5. Classification Head

A special [CLS] token is prepended to the sequence, and its final representation is used for classification.

## 6. Summary

ViT demonstrates that pure Transformer architectures can achieve competitive results on vision tasks, given sufficient data and compute.

For more, see the original paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) 