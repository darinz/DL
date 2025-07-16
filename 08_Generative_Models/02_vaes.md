# Variational Autoencoders (VAEs)

## 1. Introduction

Variational Autoencoders (VAEs) are probabilistic generative models that learn to encode data into a latent space and decode samples from this space back to data. Unlike traditional autoencoders, VAEs impose a probabilistic structure on the latent space, enabling generative capabilities.

> **Key Insight:** VAEs bridge the gap between deep learning and probabilistic graphical models, allowing us to generate new data by sampling from a learned latent distribution.

> **Did you know?** VAEs can be used for image generation, denoising, anomaly detection, and even semi-supervised learning!

## 2. Mathematical Formulation

The core objective of a VAE is to maximize the likelihood of the data $`p(x)`$. However, directly maximizing $`\log p(x)`$ is intractable due to the integral over latent variables $`z`$. Instead, VAEs maximize the Evidence Lower Bound (ELBO) on the data log-likelihood:

```math
\log p(x) \geq \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))
```

- $`q_\phi(z|x)`$: Encoder (approximate posterior)
- $`p_\theta(x|z)`$: Decoder (likelihood)
- $`p(z)`$: Prior (usually standard normal)
- $`D_{\mathrm{KL}}`$: Kullback-Leibler divergence

### Step-by-Step Derivation

1. **Marginal Likelihood:**
   $`p(x) = \int p(x|z) p(z) dz`$
2. **Introduce Approximate Posterior:**
   $`\log p(x) = \log \int p(x|z) p(z) dz = \log \int q_\phi(z|x) \frac{p(x|z)p(z)}{q_\phi(z|x)} dz`$
3. **Apply Jensen's Inequality:**
   $`\log p(x) \geq \mathbb{E}_{q_\phi(z|x)} \left[ \log \frac{p(x|z)p(z)}{q_\phi(z|x)} \right]`$
4. **Rearrange:**
   $`= \mathbb{E}_{q_\phi(z|x)} [\log p(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))`$

> **Geometric Intuition:** The KL divergence term encourages the learned latent distribution to stay close to the prior, while the reconstruction term ensures the latent code captures meaningful information about $`x`$.

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
        h = torch.relu(self.fc1(x))  # Nonlinear transformation
        mu = self.fc_mu(h)           # Mean of q(z|x)
        logvar = self.fc_logvar(h)   # Log-variance of q(z|x)
        return mu, logvar
```
*The encoder outputs the parameters of a Gaussian distribution for each input $`x`$.*

### Decoder
```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
    def forward(self, z):
        h = torch.relu(self.fc1(z))      # Nonlinear transformation
        return torch.sigmoid(self.fc2(h)) # Output in [0,1] for binary data
```
*The decoder reconstructs the input from the latent code $`z`$.*

> **Try it yourself!** Modify the encoder/decoder hidden size or add more layers. How does this affect the quality of generated samples?

## 4. Training Process

The loss function combines reconstruction loss and KL divergence:

```math
\mathcal{L}(x) = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))
```

- **Reconstruction Loss:** Measures how well the decoder reconstructs $`x`$ from $`z`$ (often binary cross-entropy or MSE).
- **KL Divergence:** Regularizes the latent space to match the prior $`p(z)`$.

### Reparameterization Trick
To backpropagate through stochastic nodes, use:
```python
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)   # Convert log-variance to std
    eps = torch.randn_like(std)     # Sample from standard normal
    return mu + eps * std           # Sample from N(mu, std^2)
```
*This trick allows gradients to flow through the sampling operation!*

> **Common Pitfall:** Forgetting the reparameterization trick will prevent the model from learning via backpropagation.

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
*Each step: encode $`x \to (\mu, \log \sigma^2)`$, sample $`z`$, decode $`z \to \hat{x}`$, compute losses, and update parameters.*

> **Try it yourself!** Visualize the learned latent space by projecting $`z`$ to 2D (e.g., using t-SNE or PCA). What patterns do you observe?

## 5. Common Issues and Solutions
- **Posterior collapse:** Decoder ignores latent code. Solution: Use KL annealing or $`\beta`$-VAE (increase KL weight gradually).
- **Blurry samples:** Use more expressive decoders or alternative likelihoods (e.g., discretized logistic, pixelCNN).

> **Key Insight:** The balance between reconstruction and KL loss is crucial. Too much KL regularization can lead to uninformative latent codes.

---

For more details, see the main README and referenced papers. 