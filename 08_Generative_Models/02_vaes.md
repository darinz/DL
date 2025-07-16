# Variational Autoencoders (VAEs)

## 1. Introduction

Variational Autoencoders (VAEs) are probabilistic generative models that learn to encode data into a latent space and decode samples from this space back to data. Unlike traditional autoencoders, VAEs impose a probabilistic structure on the latent space, enabling generative capabilities.

## 2. Mathematical Formulation

VAEs maximize the Evidence Lower Bound (ELBO) on the data log-likelihood:

```math
\log p(x) \geq \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))
```

- $`q_\phi(z|x)`$: Encoder (approximate posterior)
- $`p_\theta(x|z)`$: Decoder (likelihood)
- $`p(z)`$: Prior (usually standard normal)
- $`D_{\mathrm{KL}}`$: Kullback-Leibler divergence

## 3. Encoder and Decoder Architecture

The encoder maps input $`x`$ to a distribution over latent variables $`z`$. The decoder reconstructs $`x`$ from $`z`$.

### Encoder
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

### Decoder
```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))
```

## 4. Training Process

The loss function combines reconstruction loss and KL divergence:

```math
\mathcal{L}(x) = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))
```

### Reparameterization Trick
To backpropagate through stochastic nodes, use:
```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

### VAE Training Loop (PyTorch)
```python
for epoch in range(num_epochs):
    for x in dataloader:
        mu, logvar = encoder(x)
        z = reparameterize(mu, logvar)
        x_recon = decoder(z)
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 5. Common Issues and Solutions
- **Posterior collapse:** Decoder ignores latent code. Solution: Use KL annealing or $`\eta`$-VAE.
- **Blurry samples:** Use more expressive decoders or alternative likelihoods.

---

For more details, see the main README and referenced papers. 