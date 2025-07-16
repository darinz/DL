# Flow-based Models

## 1. Introduction

Flow-based models are generative models that use invertible transformations to map data to latent variables, allowing exact likelihood computation and efficient sampling.

> **Explanation:**
> Flow-based models are a class of generative models that transform data into a simple distribution (like a Gaussian) using a series of invertible functions. Because these functions are invertible, you can go both ways: generate new data by sampling from the simple distribution and mapping it back, or compute the likelihood of data exactly.

> **Key Insight:** Unlike VAEs and GANs, flow-based models provide exact log-likelihoods and allow both sampling and inference to be efficient and tractable.

> **Did you know?** Flow-based models can generate high-quality images and are used in tasks like super-resolution and audio synthesis!

## 2. Change of Variables Formula

The log-likelihood of data $`x`$ under a flow-based model is given by the change of variables formula:

```math
\log p(x) = \log p(z) + \sum_{i=1}^K \log \left| \det \frac{\partial f_i}{\partial h_{i-1}} \right|
```

> **Math Breakdown:**
> - $`p(z)`$: Probability of the latent variable $`z`$ (usually standard normal).
> - $`f_i`$: Each invertible transformation in the sequence.
> - $`h_{i-1}`$: The output of the previous transformation (or the input for $`i=1`$).
> - $`\det \frac{\partial f_i}{\partial h_{i-1}}`$: The determinant of the Jacobian matrix, which measures how the transformation stretches or compresses space.
> - The sum over $`i`$ accumulates the effect of all transformations.

where $`x = f_K \circ \cdots \circ f_1(z)`$ and each $`f_i`$ is invertible.

### Step-by-Step Derivation
1. **Start with a latent variable** $`z`$ sampled from a simple distribution (e.g., standard normal).
   > **Explanation:**
   > The model begins with a simple, known distribution for $`z`$.
2. **Apply a sequence of invertible transformations** $`f_1, ..., f_K`$ to obtain $`x`$.
   > **Explanation:**
   > Each transformation warps the latent space, gradually shaping it to match the data distribution.
3. **Compute the log-likelihood** using the change of variables formula above.
   > **Math Breakdown:**
   > The log-likelihood of the data is the sum of the log-probability of the latent variable and the log-determinants of the Jacobians of each transformation.

> **Geometric Intuition:** Each transformation $`f_i`$ "warps" the latent space, gradually shaping it to match the data distribution.

## 3. Invertible Transformations

Each transformation must be invertible and have a tractable Jacobian determinant. This ensures we can compute $`p(x)`$ exactly and sample efficiently.

> **Explanation:**
> If a transformation is not invertible, we can't go back and forth between data and latent space. If the Jacobian is hard to compute, we can't evaluate the likelihood efficiently. Special architectures like coupling layers are designed to solve these problems.

> **Common Pitfall:** If the Jacobian determinant is hard to compute, training and sampling become infeasible. That's why special architectures (like coupling layers) are used.

## 4. RealNVP

RealNVP uses coupling layers for efficient computation of the Jacobian determinant. In a coupling layer, only part of the input is transformed at a time, making the Jacobian triangular and easy to compute.

### Coupling Layer Example (PyTorch)
```python
import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Linear(dim // 2, dim // 2)
        self.translate = nn.Linear(dim // 2, dim // 2)
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # Split input into two halves
        s = self.scale(x1)          # Scale function
        t = self.translate(x1)      # Translation function
        y1 = x1                     # y1 is unchanged
        y2 = x2 * torch.exp(s) + t  # y2 is transformed
        return torch.cat([y1, y2], dim=1)
```
> **Code Walkthrough:**
> - The input $`x`$ is split into two halves: $`x_1`$ and $`x_2`$.
> - $`x_1`$ is left unchanged, while $`x_2`$ is scaled and translated based on $`x_1`$.
> - The scale and translation are learned by neural networks.
> - This design makes the transformation invertible and the Jacobian easy to compute (triangular matrix).

*The coupling layer only transforms half the variables at a time, making the transformation invertible and the Jacobian easy to compute!*

> **Try it yourself!** Modify the coupling layer to swap which half is transformed at each layer. How does this affect expressiveness?

## 5. Glow

Glow introduces invertible $`1 \times 1`$ convolutions for improved expressiveness. These allow the model to permute and mix channels, increasing flexibility.

> **Explanation:**
> Invertible $`1 \times 1`$ convolutions act like a learned permutation of the input channels, making the flow more expressive than simple fixed permutations. This helps the model capture more complex dependencies in the data.

> **Key Insight:** Invertible $`1 \times 1`$ convolutions act like a learned permutation, making the flow more expressive than simple fixed permutations.

## 6. Example: Simple Flow Model (PyTorch)
```python
class SimpleFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.coupling1 = CouplingLayer(dim)
        self.coupling2 = CouplingLayer(dim)
    def forward(self, x):
        x = self.coupling1(x)  # First coupling layer
        x = self.coupling2(x)  # Second coupling layer
        return x
```
> **Code Walkthrough:**
> - This model stacks two coupling layers to increase expressiveness.
> - Each coupling layer applies an invertible transformation to the input.
> - More layers can be added to make the model more powerful.

*Stacking multiple coupling layers increases the expressiveness of the flow model.*

> **Try it yourself!** Add more coupling layers or experiment with different invertible transformations.

---

For more details, see the main README and referenced papers. 