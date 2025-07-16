# Flow-based Models

## 1. Introduction

Flow-based models are generative models that use invertible transformations to map data to latent variables, allowing exact likelihood computation and efficient sampling.

## 2. Change of Variables Formula

The log-likelihood of data $`x`$ under a flow-based model is:

```math
\log p(x) = \log p(z) + \sum_{i=1}^K \log \left| \det \frac{\partial f_i}{\partial h_{i-1}} \right|
```

where $`x = f_K \circ \cdots \circ f_1(z)`$ and each $`f_i`$ is invertible.

## 3. Invertible Transformations

Each transformation must be invertible and have a tractable Jacobian determinant.

## 4. RealNVP

RealNVP uses coupling layers for efficient computation of the Jacobian determinant.

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
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale(x1)
        t = self.translate(x1)
        y1 = x1
        y2 = x2 * torch.exp(s) + t
        return torch.cat([y1, y2], dim=1)
```

## 5. Glow

Glow introduces invertible $1 \times 1$ convolutions for improved expressiveness.

## 6. Example: Simple Flow Model (PyTorch)
```python
class SimpleFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.coupling1 = CouplingLayer(dim)
        self.coupling2 = CouplingLayer(dim)
    def forward(self, x):
        x = self.coupling1(x)
        x = self.coupling2(x)
        return x
```

---

For more details, see the main README and referenced papers. 