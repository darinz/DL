# Generative Adversarial Networks (GANs)

## 1. Introduction

Generative Adversarial Networks (GANs) are a class of generative models introduced by Ian Goodfellow in 2014. GANs consist of two neural networks, a **generator** and a **discriminator**, which are trained in opposition to each other. The generator tries to create data that mimics the real data distribution, while the discriminator tries to distinguish between real and generated data.

## 2. Mathematical Formulation

The GAN framework is formulated as a minimax game between the generator $`G`$ and the discriminator $`D`$:

```math
\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]
```

- $`p_{\text{data}}(x)`$: Distribution of real data
- $`p_z(z)`$: Prior distribution (e.g., Gaussian noise)
- $`G(z)`$: Generator output
- $`D(x)`$: Probability that $`x`$ is real

## 3. Training Process

Training alternates between updating the discriminator and the generator:

1. **Discriminator step:**
   - Maximize the probability of assigning the correct label to both real and generated samples.
2. **Generator step:**
   - Minimize $`\log(1 - D(G(z)))`$ (or maximize $`\log D(G(z))`$ for better gradients).

### Pseudocode
```python
# Pseudocode for GAN training loop
for epoch in range(num_epochs):
    for real_data in dataloader:
        # Train Discriminator
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z)
        loss_D = -torch.mean(torch.log(D(real_data)) + torch.log(1 - D(fake_data)))
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z)
        loss_G = -torch.mean(torch.log(D(fake_data)))
        loss_G.backward()
        optimizer_G.step()
```

## 4. Common Issues in GAN Training
- **Mode collapse:** Generator produces limited variety.
- **Non-convergence:** Oscillating or diverging losses.
- **Vanishing gradients:** Discriminator becomes too strong.

### Solutions
- Use alternative loss functions (e.g., Wasserstein GAN)
- Feature matching, minibatch discrimination
- Careful architecture and hyperparameter choices

## 5. DCGAN (Deep Convolutional GAN)

DCGANs use convolutional layers for both the generator and discriminator, making them well-suited for image generation tasks.

### DCGAN Generator Example (PyTorch)
```python
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_maps):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)
```

## 6. StyleGAN

StyleGAN introduces a style-based generator architecture, enabling control over image features at different levels. It uses adaptive instance normalization (AdaIN) and mapping networks.

- **Key idea:** Separate the latent space from the image synthesis process for better disentanglement.

## 7. CycleGAN

CycleGAN enables image-to-image translation without paired data by introducing a cycle-consistency loss:

```math
\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_{\text{data}}(x)} [\| F(G(x)) - x \|_1] + \mathbb{E}_{y \sim p_{\text{data}}(y)} [\| G(F(y)) - y \|_1]
```

- $`G`$: Maps domain X to Y
- $`F`$: Maps domain Y to X

## 8. Conditional GANs (cGANs)

Conditional GANs incorporate label information into both the generator and discriminator, enabling class-conditional generation.

```math
\min_G \max_D \; \mathbb{E}_{x, y \sim p_{\text{data}}(x, y)} [\log D(x, y)] + \mathbb{E}_{z \sim p_z(z), y \sim p_y(y)} [\log(1 - D(G(z, y), y))]
```

### cGAN Example (PyTorch)
```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        return img.view(img.size(0), *img_shape)
```

---

For more details and advanced topics, see the references and further reading sections in the main README. 