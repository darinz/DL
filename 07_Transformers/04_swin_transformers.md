# Swin Transformers

> **Key Insight:** Swin Transformers bring the hierarchical, multi-scale processing of CNNs to the Transformer world, enabling efficient and scalable vision models.

Swin Transformers introduce a hierarchical vision Transformer using shifted windows, enabling scalable and efficient modeling of large images.

## 1. Overview

Swin Transformers build feature maps at multiple scales, similar to CNNs, and use local self-attention within non-overlapping windows.

> **Did you know?** The name "Swin" comes from "Shifted Windows," the core innovation that enables cross-window information flow.

## 2. Window-based Self-Attention

Instead of global self-attention, Swin computes self-attention within local windows, reducing computational complexity from $`O((HW)^2)`$ to $`O(M^2HW/M^2) = O(HW M^2)`$, where $`M`$ is the window size.

> **Explanation:**
> By restricting self-attention to local windows, Swin Transformers dramatically reduce the number of computations compared to global attention. This makes them much more efficient for large images.

#### Geometric/Visual Explanation

Imagine dividing an image into small windows and letting each window process its own content independently. This is much more efficient than having every patch attend to every other patch in the image.

> **Common Pitfall:**
> Using only local windows can limit the model's ability to capture long-range dependencies—hence the need for shifted windows.

## 3. Shifted Window Mechanism

To allow cross-window connections, the window partitioning is shifted between layers. This means that in one layer, windows are placed at certain positions, and in the next, they are shifted so that different patches are grouped together.

> **Explanation:**
> Shifting the windows ensures that information can flow between neighboring windows, enabling the model to capture both local and global context over multiple layers.

#### Intuitive Explanation

Shifting the windows ensures that information can flow between neighboring windows, enabling the model to capture both local and global context over multiple layers.

## 4. Hierarchical Representation

Patch merging layers reduce the number of tokens and increase feature dimension, building a hierarchical representation.

| Stage                | Operation                | Effect                                 |
|----------------------|-------------------------|----------------------------------------|
| Patch embedding      | Split image into patches | Initial tokenization                   |
| Window attention     | Local self-attention     | Efficient local context modeling       |
| Shifted windows      | Shift window positions   | Cross-window information flow          |
| Patch merging        | Merge patches            | Build multi-scale, hierarchical features|

> **Explanation:**
> The hierarchical design allows Swin Transformers to process images at multiple scales, similar to how CNNs use pooling to build up from local to global features.

> **Did you know?** This hierarchical design allows Swin Transformers to be used as a backbone for tasks like object detection and segmentation, not just classification.

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

> **Code Walkthrough:**
> - The window partitioning function rearranges the image tensor so that each window can be processed independently by self-attention.
> - Changing the window size changes the number and shape of windows, affecting the model's efficiency and receptive field.

> **Try it yourself!** Change the window size and see how the number and shape of windows change for a fixed image size.

## 6. Summary & Next Steps

Swin Transformers achieve state-of-the-art results on various vision benchmarks and are highly efficient for large-scale image processing.

| Component           | Role in Swin Transformer                |
|---------------------|----------------------------------------|
| Window attention    | Efficient local self-attention          |
| Shifted windows     | Enables cross-window context            |
| Patch merging       | Builds hierarchical, multi-scale features|
| Hierarchical design | Scalable to large images and tasks      |

> **Key Insight:** Swin Transformers combine the best of CNNs (hierarchy, locality) and Transformers (flexible attention) for vision.

### Next Steps
- Implement a simple Swin Transformer block and visualize window partitioning.
- Explore how shifting windows affects information flow.
- Read the original paper: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)

> **Did you know?** Swin Transformers are now used as backbones in many state-of-the-art vision models, including object detection and segmentation systems. 