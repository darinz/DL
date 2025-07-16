# Training Techniques

This section covers essential techniques for training deep neural networks effectively, including regularization methods, normalization techniques, and initialization strategies.

## Table of Contents

1. [Regularization Methods](#regularization-methods)
2. [Normalization Techniques](#normalization-techniques)
3. [Initialization Strategies](#initialization-strategies)

## Regularization Methods

Regularization techniques help prevent overfitting and improve the generalization ability of neural networks.

### Dropout

Dropout randomly deactivates neurons during training to prevent co-adaptation and improve generalization.

**Mathematical Formulation:**
```math
\text{Dropout}(x, p) = \begin{cases}
\frac{x}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}
```

**Key Properties:**
- During training: Randomly zeroes activations with probability $`p`$
- During inference: Scales activations by $`1-p`$ to maintain expected values
- Effective dropout rates: 0.2-0.5 for hidden layers, 0.5-0.7 for input layers

### Weight Decay (L2 Regularization)

Adds a penalty term to the loss function to discourage large weights.

**Mathematical Formulation:**
```math
L_{\text{total}} = L_{\text{original}} + \frac{\lambda}{2} \sum_{i} w_i^2
```

**Gradient Update:**
```math
\frac{\partial L_{\text{total}}}{\partial w_i} = \frac{\partial L_{\text{original}}}{\partial w_i} + \lambda w_i
```

**Weight Update Rule:**
```math
w_i \leftarrow w_i - \alpha \left(\frac{\partial L_{\text{original}}}{\partial w_i} + \lambda w_i\right) = (1 - \alpha\lambda)w_i - \alpha\frac{\partial L_{\text{original}}}{\partial w_i}
```

### Early Stopping

Monitors validation performance and stops training when overfitting begins.

**Implementation Strategy:**
- Track validation loss/metric over epochs
- Stop when validation performance doesn't improve for $`k`$ epochs
- Save best model weights during training

### Data Augmentation

Expands training data through transformations to improve generalization.

**Common Techniques:**
- **Image Data:** Rotation, scaling, flipping, color jittering, cropping
- **Text Data:** Synonym replacement, back-translation, random insertion/deletion
- **Audio Data:** Time stretching, pitch shifting, noise injection

## Normalization Techniques

Normalization techniques stabilize training by controlling the distribution of activations and gradients.

### Batch Normalization (BatchNorm)

Normalizes layer inputs across the batch dimension.

**Mathematical Formulation:**
```math
\text{BN}(x) = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
```

Where:
- $`\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i`$ (batch mean)
- $`\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2`$ (batch variance)
- $`\gamma, \beta`$ are learnable parameters
- $`\epsilon`$ is a small constant for numerical stability

**Training vs Inference:**
- **Training:** Uses batch statistics
- **Inference:** Uses running averages of batch statistics

### Layer Normalization

Normalizes across the feature dimension for each sample independently.

**Mathematical Formulation:**
```math
\text{LN}(x) = \gamma \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta
```

Where:
- $`\mu_L = \frac{1}{d}\sum_{i=1}^{d} x_i`$ (layer mean)
- $`\sigma_L^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu_L)^2`$ (layer variance)
- $`d`$ is the feature dimension

### Instance Normalization

Normalizes each individual sample across spatial dimensions.

**Mathematical Formulation:**
```math
\text{IN}(x) = \gamma \frac{x - \mu_I}{\sqrt{\sigma_I^2 + \epsilon}} + \beta
```

Where:
- $`\mu_I = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{h,w}`$ (instance mean)
- $`\sigma_I^2 = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} (x_{h,w} - \mu_I)^2`$ (instance variance)

### Group Normalization

Normalizes within groups of channels.

**Mathematical Formulation:**
```math
\text{GN}(x) = \gamma \frac{x - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} + \beta
```

Where normalization is performed within each group of $`G`$ channels.

## Initialization Strategies

Proper weight initialization is crucial for training deep networks effectively.

### Xavier/Glorot Initialization

Designed for sigmoid and tanh activation functions.

**Mathematical Formulation:**
```math
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
```

Or for uniform distribution:
```math
W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)
```

**Intuition:** Maintains variance of activations and gradients across layers.

### He Initialization

Optimized for ReLU activation functions.

**Mathematical Formulation:**
```math
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
```

Or for uniform distribution:
```math
W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right)
```

**Intuition:** Accounts for ReLU's zeroing of negative activations.

### Orthogonal Initialization

Initializes weights as orthogonal matrices to preserve gradient flow.

**Mathematical Formulation:**
```math
W = U \Sigma V^T
```

Where $`U`$ and $`V`$ are orthogonal matrices, and $`\Sigma`$ contains singular values.

**Properties:**
- Preserves gradient magnitude
- Useful for recurrent neural networks
- Helps with vanishing/exploding gradients

### Pre-trained Weights

Initializes networks with weights from pre-trained models.

**Transfer Learning Approaches:**
- **Feature Extraction:** Freeze pre-trained layers, train only new layers
- **Fine-tuning:** Update all layers with smaller learning rate
- **Progressive Unfreezing:** Gradually unfreeze layers during training

**Benefits:**
- Faster convergence
- Better performance with limited data
- Leverages knowledge from large datasets

## Practical Considerations

### Choosing Regularization Techniques

- **Dropout:** Effective for fully connected layers, less effective for convolutional layers
- **Weight Decay:** Works well with most architectures, requires tuning of $`\lambda`$
- **Early Stopping:** Universal technique, requires validation set
- **Data Augmentation:** Domain-specific, should preserve semantic meaning

### Normalization Selection

- **BatchNorm:** Most effective for CNNs, requires sufficient batch size
- **LayerNorm:** Good for RNNs and transformers, independent of batch size
- **InstanceNorm:** Effective for style transfer and image generation
- **GroupNorm:** Alternative to BatchNorm for small batch sizes

### Initialization Guidelines

- **Xavier/Glorot:** Use with sigmoid/tanh activations
- **He:** Use with ReLU and variants
- **Orthogonal:** Use for RNNs and transformers
- **Pre-trained:** Use when available and applicable

## References

- Srivastava, N., et al. "Dropout: A simple way to prevent neural networks from overfitting"
- Ioffe, S., & Szegedy, C. "Batch normalization: Accelerating deep network training by reducing internal covariate shift"
- Ba, J. L., et al. "Layer normalization"
- Glorot, X., & Bengio, Y. "Understanding the difficulty of training deep feedforward neural networks"
- He, K., et al. "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" 