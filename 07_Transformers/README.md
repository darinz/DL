# Transformers & Attention

[![Transformers](https://img.shields.io/badge/Transformers-Attention%20Mechanisms-blue?style=for-the-badge&logo=brain)](https://github.com/yourusername/DL)
[![Self Attention](https://img.shields.io/badge/Self%20Attention-Multi%20Head-green?style=for-the-badge&logo=eye)](https://github.com/yourusername/DL/tree/main/07_Transformers)
[![BERT](https://img.shields.io/badge/BERT-Bidirectional-orange?style=for-the-badge&logo=google)](https://github.com/yourusername/DL/tree/main/07_Transformers)
[![GPT](https://img.shields.io/badge/GPT-Generative-purple?style=for-the-badge&logo=openai)](https://github.com/yourusername/DL/tree/main/07_Transformers)
[![T5](https://img.shields.io/badge/T5-Text%20to%20Text-red?style=for-the-badge&logo=google)](https://github.com/yourusername/DL/tree/main/07_Transformers)
[![ViT](https://img.shields.io/badge/ViT-Vision%20Transformers-yellow?style=for-the-badge&logo=image)](https://github.com/yourusername/DL/tree/main/07_Transformers)
[![Swin](https://img.shields.io/badge/Swin-Hierarchical-blue?style=for-the-badge&logo=window-maximize)](https://github.com/yourusername/DL/tree/main/07_Transformers)
[![LLMs](https://img.shields.io/badge/LLMs-Large%20Language-orange?style=for-the-badge&logo=language)](https://github.com/yourusername/DL/tree/main/07_Transformers)

This module covers the foundational concepts and architectures in the field of Transformers and Attention mechanisms, which have revolutionized deep learning for both natural language processing and computer vision.

## 1. [Transformer Architecture](01_transformer_architecture.md)

The Transformer architecture, introduced by Vaswani et al. (2017), is based on self-attention mechanisms and dispenses with recurrence and convolutions entirely. The core idea is to allow each position in the input sequence to attend to all other positions, enabling efficient modeling of long-range dependencies.

### Self-Attention Mechanism
Given an input sequence of vectors $`X = [x_1, x_2, \ldots, x_n]`$, the self-attention mechanism computes queries $`Q`$, keys $`K`$, and values $`V`$ as linear projections:

```math
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
```

The attention scores are computed as:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

where $`d_k`$ is the dimension of the key vectors.

### Multi-Head Attention
Multiple attention heads allow the model to jointly attend to information from different representation subspaces:

```math
\text{MultiHead}(Q, K, V) = [\text{head}_1; \ldots; \text{head}_h]W^O
```

where each $`\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`$.

## 2. [Large Language Models: BERT, GPT, T5](02_large_language_models.md)

- **BERT (Bidirectional Encoder Representations from Transformers):**
  - Uses only the encoder part of the Transformer.
  - Trained with masked language modeling and next sentence prediction.
  - Enables bidirectional context understanding.

- **GPT (Generative Pre-trained Transformer):**
  - Uses only the decoder part of the Transformer.
  - Trained with left-to-right language modeling.
  - Suited for text generation tasks.

- **T5 (Text-to-Text Transfer Transformer):**
  - Treats every NLP problem as a text-to-text task.
  - Uses an encoder-decoder architecture.

## 3. [Vision Transformers (ViT)](03_vision_transformers.md)

Vision Transformers adapt the Transformer architecture for image data. Images are split into patches, each patch is linearly embedded, and position embeddings are added. The sequence of patch embeddings is then processed by a standard Transformer encoder.

### Patch Embedding
Given an image $`x \in \mathbb{R}^{H \times W \times C}`$, it is split into $`N`$ patches of size $`P \times P`$:

```math
N = \frac{HW}{P^2}
```

Each patch is flattened and projected to a $`D`$-dimensional embedding.

## 4. [Swin Transformers](04_swin_transformers.md)

Swin Transformers introduce a hierarchical architecture for vision tasks, using shifted windows for self-attention. This enables scalable modeling of large images and efficient computation.

- **Hierarchical Representation:** Builds feature maps at multiple scales.
- **Shifted Window Attention:** Computes self-attention within local windows, with window positions shifted between layers to allow cross-window connections.

---

This module provides the theoretical foundation and practical insights into the most influential Transformer-based models in deep learning. 