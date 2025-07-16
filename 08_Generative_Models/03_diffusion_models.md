# Diffusion Models

## 1. Introduction

Diffusion models are generative models that learn to generate data by reversing a gradual noising process. They have achieved state-of-the-art results in image and audio generation.

> **Key Insight:** Diffusion models generate data by simulating a physical process: gradually adding noise to data and then learning to reverse this process step by step.

> **Did you know?** The "diffusion" in these models is inspired by thermodynamics and statistical physics, where particles spread out over time due to random motion.

## 2. Forward and Reverse Processes

The forward process gradually adds noise to data, transforming a clean sample $`x_0`$ into pure noise $`x_T`$ over $`T`$ steps:

```math
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})
```

- $`\beta_t`$: Variance schedule (controls how much noise is added at each step)
- $`x_t`$: Noisy data at step $`t`$

The reverse process learns to denoise, i.e., to recover $`x_{t-1}`$ from $`x_t`$:

```math
p_\theta(x_{t-1} | x_t)
```

> **Geometric Intuition:** Imagine a photo slowly dissolving into static. The model learns to "run the movie backwards" and reconstruct the original image from noise.

## 3. Mathematical Formulation

The model is trained to predict the noise added at each step. The typical loss function is:

```math
\mathcal{L}_t = \mathbb{E}_{x, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
```

- $`x_t`$: Noisy data at step $`t`$
- $`\epsilon`$: True noise
- $`\epsilon_\theta`$: Predicted noise by the neural network

### Step-by-Step Breakdown
1. **Sample a clean data point** $`x_0`$.
2. **Sample a timestep** $`t`$ uniformly from $`1, ..., T`$.
3. **Add noise**: $`x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon`$, where $`\epsilon \sim \mathcal{N}(0, I)`$ and $`\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)`$.
4. **Train the model** to predict $`\epsilon`$ from $`x_t`$ and $`t`$.

> **Common Pitfall:** Forgetting to use the correct noise schedule $`\beta_t`$ can lead to poor sample quality or training instability.

## 4. DDPM (Denoising Diffusion Probabilistic Models)

DDPMs use a Markov chain to add and remove noise. The model is trained to reverse this process, step by step.

### PyTorch Example: Forward Process
```python
import torch

def q_sample(x_0, t, betas):
    """
    Adds noise to x_0 at timestep t using the variance schedule betas.
    x_0: original data
    t: timestep (int or tensor)
    betas: 1D tensor of noise schedule
    """
    noise = torch.randn_like(x_0)  # Gaussian noise
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1 - betas, dim=0))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))
    # Linearly interpolate between x_0 and noise
    return sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise
```
*This function simulates the forward (noising) process for a given timestep $`t`$.*

> **Try it yourself!** Visualize $`x_t`$ for different $`t`$ values. How does the image change as more noise is added?

## 5. DDIM (Denoising Diffusion Implicit Models)

DDIMs provide a deterministic variant for faster sampling. Instead of sampling at each step, DDIMs use a fixed transformation, reducing the number of steps needed to generate high-quality samples.

> **Key Insight:** DDIMs show that the stochasticity in the reverse process is not always necessary for good sample quality.

## 6. Stable Diffusion

Stable Diffusion is a latent diffusion model that operates in a compressed latent space for efficient high-resolution image synthesis. By working in a lower-dimensional space, it enables fast and memory-efficient generation of large images.

> **Did you know?** Stable Diffusion powers many modern text-to-image applications and is open-source!

## 7. Example: Denoising Network (PyTorch)
```python
import torch.nn as nn

class DenoiseNet(nn.Module):
    def __init__(self, img_channels, base_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, base_channels, 3, 1, 1),  # First convolution
            nn.ReLU(),
            nn.Conv2d(base_channels, img_channels, 3, 1, 1)   # Output layer
        )
    def forward(self, x, t):
        # t can be used for time embedding (not shown here)
        return self.net(x)
```
*This simple network predicts the noise $`\epsilon`$ given a noisy image $`x_t`$ and (optionally) the timestep $`t`$.*

> **Try it yourself!** Add a time embedding to the network and see how it affects performance.

---

For more details, see the main README and referenced papers. 