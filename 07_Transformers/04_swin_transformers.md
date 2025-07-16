# Swin Transformers

Swin Transformers introduce a hierarchical vision Transformer using shifted windows, enabling scalable and efficient modeling of large images.

## 1. Overview

Swin Transformers build feature maps at multiple scales, similar to CNNs, and use local self-attention within non-overlapping windows.

## 2. Window-based Self-Attention

Instead of global self-attention, Swin computes self-attention within local windows, reducing computational complexity from $`O((HW)^2)`$ to $`O(M^2HW/M^2) = O(HW M^2)`$, where $`M`$ is the window size.

## 3. Shifted Window Mechanism

To allow cross-window connections, the window partitioning is shifted between layers.

## 4. Hierarchical Representation

Patch merging layers reduce the number of tokens and increase feature dimension, building a hierarchical representation.

## 5. Python Example: Window Partitioning
```python
import torch

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

# Example usage
x = torch.randn(2, 8, 8, 96)  # (B, H, W, C)
windows = window_partition(x, 4)
print(windows.shape)  # (8, 4, 4, 96)
```

## 6. Summary

Swin Transformers achieve state-of-the-art results on various vision benchmarks and are highly efficient for large-scale image processing.

For more, see the original paper: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) 