# Generative Models

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

## 2. Variational Autoencoders (VAEs)

VAEs are probabilistic generative models that learn a latent variable model for data. They encode data into a latent space and decode samples from this space back to data.

**Objective (ELBO):**
```math
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))
```
where $`q_\phi(z|x)`$ is the encoder, $`p_\theta(x|z)`$ is the decoder, and $`p(z)`$ is the prior.

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

## 4. Flow-based Models

Flow-based models use invertible transformations to map data to latent variables, allowing exact likelihood computation and efficient sampling.

- **RealNVP**: Uses coupling layers for tractable Jacobian determinants.
- **Glow**: Introduces invertible $1 \times 1$ convolutions for improved expressiveness.

**Change of Variables Formula:**
```math
\log p(x) = \log p(z) + \sum_{i=1}^K \log \left| \det \frac{\partial f_i}{\partial h_{i-1}} \right|
```
where $`x = f_K \circ \cdots \circ f_1(z)`$.

---

For further reading, see the main [README](../README.md) and the referenced papers for each model. 