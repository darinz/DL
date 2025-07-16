# Generative Models

[![Generative Models](https://img.shields.io/badge/Generative%20Models-Data%20Generation-blue?style=for-the-badge&logo=brain)](https://github.com/yourusername/DL)
[![GANs](https://img.shields.io/badge/GANs-Adversarial-green?style=for-the-badge)](https://github.com/yourusername/DL/tree/main/08_Generative_Models)
[![VAEs](https://img.shields.io/badge/VAEs-Variational-orange?style=for-the-badge)](https://github.com/yourusername/DL/tree/main/08_Generative_Models)
[![Diffusion](https://img.shields.io/badge/Diffusion-Denoising-purple?style=for-the-badge)](https://github.com/yourusername/DL/tree/main/08_Generative_Models)
[![Flow Models](https://img.shields.io/badge/Flow%20Models-Invertible-red?style=for-the-badge)](https://github.com/yourusername/DL/tree/main/08_Generative_Models)
[![StyleGAN](https://img.shields.io/badge/StyleGAN-Style%20Based-yellow?style=for-the-badge)](https://github.com/yourusername/DL/tree/main/08_Generative_Models)
[![Stable Diffusion](https://img.shields.io/badge/Stable%20Diffusion-Latent-blue?style=for-the-badge)](https://github.com/yourusername/DL/tree/main/08_Generative_Models)
[![Image Synthesis](https://img.shields.io/badge/Image%20Synthesis-Generation-orange?style=for-the-badge)](https://github.com/yourusername/DL/tree/main/08_Generative_Models)

Generative models are a class of machine learning models that learn to model the underlying distribution of data, enabling them to generate new, similar samples. They are fundamental in unsupervised and self-supervised learning, and have applications in image synthesis, data augmentation, and more.

## 1. Generative Adversarial Networks (GANs)

GANs consist of two neural networks, a **generator** $`G`$ and a **discriminator** $`D`$, which are trained simultaneously via adversarial processes. The generator tries to produce data resembling the real data, while the discriminator tries to distinguish between real and generated data.

**Objective:**
```math
\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log(1 - D(G(z)))]
```

### Variants
- **DCGAN**: Deep Convolutional GAN, uses convolutional layers for image generation.
- **StyleGAN**: Introduces style-based generator architecture for high-quality image synthesis.
- **CycleGAN**: Enables image-to-image translation without paired data using cycle-consistency loss.
- **Conditional GANs**: Incorporate label information to generate class-conditional samples.

[Read the comprehensive GANs guide &rarr;](01_gans.md)

## 2. Variational Autoencoders (VAEs)

VAEs are probabilistic generative models that learn a latent variable model for data. They encode data into a latent space and decode samples from this space back to data.

**Objective (ELBO):**
```math
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))
```
where $`q_\phi(z|x)`$ is the encoder, $`p_\theta(x|z)`$ is the decoder, and $`p(z)`$ is the prior.

[Read the comprehensive VAEs guide &rarr;](02_vaes.md)

## 3. Diffusion Models

Diffusion models generate data by reversing a gradual noising process. They have achieved state-of-the-art results in image and audio generation.

- **DDPM (Denoising Diffusion Probabilistic Models)**: Learn to reverse a Markov chain of noise-adding steps.
- **DDIM (Denoising Diffusion Implicit Models)**: Deterministic variant for faster sampling.
- **Stable Diffusion**: Latent diffusion model for efficient high-resolution image synthesis.

**Forward Process:**
```math
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I})
```
**Reverse Process:**
```math
p_\theta(x_{t-1} | x_t)
```

[Read the comprehensive Diffusion Models guide &rarr;](03_diffusion_models.md)

## 4. Flow-based Models

Flow-based models use invertible transformations to map data to latent variables, allowing exact likelihood computation and efficient sampling.

- **RealNVP**: Uses coupling layers for tractable Jacobian determinants.
- **Glow**: Introduces invertible $1 \times 1$ convolutions for improved expressiveness.

**Change of Variables Formula:**
```math
\log p(x) = \log p(z) + \sum_{i=1}^K \log \left| \det \frac{\partial f_i}{\partial h_{i-1}} \right|
```
where $`x = f_K \circ \cdots \circ f_1(z)`$.

[Read the comprehensive Flow-based Models guide &rarr;](04_flow_based_models.md)

---

For further reading, see the main [README](../README.md) and the referenced papers for each model. 