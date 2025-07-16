# Vision Transformers (ViT)

> **Key Insight:** Vision Transformers (ViT) show that pure attention-based architectures can match or surpass convolutional networks on vision tasks, given enough data and compute.

Vision Transformers (ViT) adapt the Transformer architecture for image data, enabling state-of-the-art performance on computer vision tasks.

## 1. Overview

ViT splits an image into fixed-size patches, flattens them, and projects them into a sequence of embeddings, which are then processed by a standard Transformer encoder.

> **Did you know?** The original ViT paper demonstrated that, with large-scale pretraining, Transformers can outperform CNNs on ImageNet and other benchmarks.

## 2. Patch Embedding
Given an image $`x \in \mathbb{R}^{H \times W \times C}`$, it is split into $`N`$ patches of size $`P \times P`$:

```math
N = \frac{HW}{P^2}
```

Each patch is flattened and projected to a $`D`$-dimensional embedding.

#### Geometric/Visual Explanation

Imagine cutting an image into a grid of small squares (patches), flattening each square into a vector, and treating the image as a sequence of these vectors—just like words in a sentence.

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

> **Code Commentary:** The convolutional layer with stride and kernel size equal to the patch size efficiently extracts and flattens patches.

> **Try it yourself!** Change the patch size and observe how the number of patches $`N`$ changes for a fixed image size.

## 3. Position Embedding

As with NLP Transformers, ViT adds position embeddings to the patch embeddings to retain spatial information.

#### Intuitive Explanation

Position embeddings give each patch a sense of "where" it is in the image, so the model can reason about spatial relationships.

## 4. Transformer Encoder

The sequence of patch embeddings is processed by a standard Transformer encoder, as described in the Transformer Architecture guide.

> **Common Pitfall:** If you omit position embeddings, the model loses spatial awareness and performs poorly on vision tasks.

## 5. Classification Head

A special [CLS] token is prepended to the sequence, and its final representation is used for classification.

#### Geometric/Visual Explanation

The [CLS] token acts as a summary vector that aggregates information from all patches through self-attention.

## 6. Summary & Next Steps

ViT demonstrates that pure Transformer architectures can achieve competitive results on vision tasks, given sufficient data and compute.

| Component           | Role in ViT                        |
|---------------------|------------------------------------|
| Patch embedding     | Converts image to sequence         |
| Position embedding  | Encodes spatial information        |
| Transformer encoder | Models relationships between patches|
| [CLS] token         | Aggregates global information      |
| Classification head | Produces final prediction          |

> **Key Insight:** ViT bridges the gap between NLP and vision, showing that sequence models can excel in both domains.

### Next Steps
- Implement a simple ViT model and train it on a small image dataset.
- Experiment with different patch sizes and embedding dimensions.
- Read the original paper: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

> **Did you know?** Modern vision models like DeiT, Swin Transformer, and BEiT build on ViT, introducing new ideas for efficiency and scalability. 