# Masked Autoencoding

Masked autoencoding is a self-supervised learning approach where parts of the input data are masked, and the model is trained to reconstruct the missing content. This encourages the model to learn both global and local structures in the data.

## 1. Core Idea

Given an input $`x`$ and a mask $`M`$, the model receives a partially observed input $`x \odot (1 - M)`$ and is trained to predict the masked content $`x \odot M`$.

## 2. Mathematical Formulation

The typical loss for masked autoencoding is:

```math
\mathcal{L}_{\text{MAE}} = \mathbb{E}_{x, M} \left[ \| f(x \odot (1 - M)) - x \odot M \|^2 \right]
```
where:
- $`f`$ is the model (encoder + decoder)
- $`\odot`$ denotes element-wise multiplication
- $`M`$ is a binary mask

## 3. MAE (Masked Autoencoders)

MAE is a framework for self-supervised pretraining of vision transformers (ViTs) by reconstructing masked patches of images.

### MAE Pipeline
1. Randomly mask a high proportion (e.g., 75%) of image patches.
2. Encode the visible patches with a vision transformer encoder.
3. Decode the full set of patches (including masked) with a lightweight decoder.
4. Compute the reconstruction loss only on masked patches.

### MAE Example (PyTorch, simplified)
```python
import torch
import torch.nn as nn

class MAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask):
        visible = x * (1 - mask)
        encoded = self.encoder(visible)
        reconstructed = self.decoder(encoded)
        loss = ((reconstructed * mask - x * mask) ** 2).mean()
        return loss
```

## 4. SimMIM

SimMIM is another masked image modeling method, similar to MAE, but with differences in masking and reconstruction strategies. It uses a simple mean squared error loss between the reconstructed and original masked pixels.

### SimMIM Example (PyTorch, simplified)
```python
class SimMIM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask):
        masked_x = x * (1 - mask)
        features = self.encoder(masked_x)
        pred = self.decoder(features)
        loss = ((pred * mask - x * mask) ** 2).mean()
        return loss
```

## 5. Summary

Masked autoencoding enables models to learn rich representations by reconstructing missing data, which is especially effective for vision transformers and other architectures. 