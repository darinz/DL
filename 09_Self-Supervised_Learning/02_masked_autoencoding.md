# Masked Autoencoding

Masked autoencoding is a self-supervised learning approach where parts of the input data are masked, and the model is trained to reconstruct the missing content. This encourages the model to learn both global and local structures in the data.

> **Explanation:**
> Masked autoencoding teaches a model to "fill in the blanks" by hiding parts of the input and asking the model to guess what was hidden. This forces the model to understand the context and structure of the data, not just memorize details.

> **Key Insight:** By hiding parts of the input, masked autoencoding forces the model to understand context and structure, not just memorize details.

> **Did you know?** Masked autoencoding is inspired by techniques in natural language processing, like BERT's masked language modeling!

## 1. Core Idea

Given an input $`x`$ and a mask $`M`$, the model receives a partially observed input $`x \odot (1 - M)`$ and is trained to predict the masked content $`x \odot M`$.

> **Geometric Intuition:** Imagine a jigsaw puzzle with missing pieces. The model learns to "fill in the blanks" using the surrounding context.

## 2. Mathematical Formulation

The typical loss for masked autoencoding is:

```math
\mathcal{L}_{\text{MAE}} = \mathbb{E}_{x, M} \left[ \| f(x \odot (1 - M)) - x \odot M \|^2 \right]
```

> **Math Breakdown:**
> - $`f`$: The model, which includes both encoder and decoder parts.
> - $`x \odot (1 - M)`$: The observed (unmasked) part of the input.
> - $`x \odot M`$: The masked (hidden) part of the input.
> - The loss is the mean squared error between the model's prediction and the true values, but only for the masked parts.
> - $`\mathbb{E}_{x, M}`$: The expectation is taken over all data samples and possible masks.

where:
- $`f`$ is the model (encoder + decoder)
- $`\odot`$ denotes element-wise multiplication
- $`M`$ is a binary mask

### Step-by-Step Breakdown
1. **Mask part of the input**: Randomly select elements or patches to hide (set to zero or a special value).
   > **Explanation:**
   > Masking can be random or structured (e.g., blocks). The goal is to hide enough information to make the task challenging but solvable.
2. **Encode the visible part**: Pass the unmasked input through the encoder.
   > **Explanation:**
   > The encoder extracts features from the visible (unmasked) data.
3. **Decode to reconstruct**: The decoder tries to predict the original input, especially the masked parts.
   > **Explanation:**
   > The decoder uses the encoded features to reconstruct the missing content.
4. **Compute loss**: Only penalize errors on the masked regions.
   > **Math Breakdown:**
   > The loss is only calculated for the masked elements, so the model focuses on reconstructing what it cannot see.

> **Common Pitfall:** If too little is masked, the task is too easy; if too much is masked, the model may not have enough context to reconstruct.

## 3. MAE (Masked Autoencoders)

MAE is a framework for self-supervised pretraining of vision transformers (ViTs) by reconstructing masked patches of images.

### MAE Pipeline
1. Randomly mask a high proportion (e.g., 75%) of image patches.
2. Encode the visible patches with a vision transformer encoder.
3. Decode the full set of patches (including masked) with a lightweight decoder.
4. Compute the reconstruction loss only on masked patches.

> **Try it yourself!** Vary the masking ratio. How does it affect the quality of learned representations?

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
        visible = x * (1 - mask)  # Masked input
        encoded = self.encoder(visible)  # Encode visible patches
        reconstructed = self.decoder(encoded)  # Decode to reconstruct all patches
        loss = ((reconstructed * mask - x * mask) ** 2).mean()  # Loss on masked patches
        return loss
```
> **Code Walkthrough:**
> - The input $`x`$ is multiplied by $`1 - mask`$ to zero out the masked regions.
> - The encoder processes only the visible parts.
> - The decoder tries to reconstruct the full input, but the loss is only computed on the masked regions.
> - The mean squared error loss encourages the model to accurately fill in the missing content.

*The MAE model learns to reconstruct only the masked patches, encouraging efficient and robust feature learning.*

## 4. SimMIM

SimMIM is another masked image modeling method, similar to MAE, but with differences in masking and reconstruction strategies. It uses a simple mean squared error loss between the reconstructed and original masked pixels.

> **Explanation:**
> SimMIM shows that even a simple approach—masking parts of the input and reconstructing them—can be very effective for learning good representations, especially when combined with strong encoders like vision transformers.

> **Key Insight:** SimMIM shows that even simple reconstruction objectives can yield strong representations when combined with effective masking.

### SimMIM Example (PyTorch, simplified)
```python
class SimMIM(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask):
        masked_x = x * (1 - mask)  # Masked input
        features = self.encoder(masked_x)  # Encode visible part
        pred = self.decoder(features)  # Predict all pixels
        loss = ((pred * mask - x * mask) ** 2).mean()  # Loss on masked pixels
        return loss
```
> **Code Walkthrough:**
> - The input is masked in the same way as in MAE.
> - The encoder and decoder are used to reconstruct the original input.
> - The loss is only computed on the masked pixels, focusing learning on the missing information.

*SimMIM's simplicity demonstrates the power of masking for self-supervised learning.*

> **Try it yourself!** Try different masking patterns (random, block, structured) and see how they affect learning.

## 5. Summary

Masked autoencoding enables models to learn rich representations by reconstructing missing data, which is especially effective for vision transformers and other architectures. 

> **Key Insight:** Masked autoencoding is a general strategy that can be applied to images, text, audio, and more! 