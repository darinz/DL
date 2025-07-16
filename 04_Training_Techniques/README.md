# Training Techniques

This section covers essential techniques for training deep neural networks effectively, including regularization methods, normalization techniques, and initialization strategies.

> **Key Insight:**
> 
> Mastering training techniques is the difference between a model that just "runs" and one that truly learns. These methods are the foundation of robust, high-performing deep learning systems.

## Table of Contents

1. [Regularization Methods](#regularization-methods) - [Detailed Guide](01_regularization_methods.md)
2. [Normalization Techniques](#normalization-techniques) - [Detailed Guide](02_normalization_techniques.md)
3. [Initialization Strategies](#initialization-strategies) - [Detailed Guide](03_initialization_strategies.md)

---

## Regularization Methods

Regularization techniques help prevent overfitting and improve the generalization ability of neural networks.

> **Did you know?**
> 
> Without regularization, deep networks can memorize the training data—leading to poor performance on new, unseen examples.

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

> **Common Pitfall:**
> 
> Using dropout during inference will degrade performance! Always turn it off at test time.

> **Try it yourself!**
> 
> Train a model with and without dropout. Compare the training and validation accuracy curves. What do you observe?

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

> **Key Insight:**
> 
> L2 regularization (weight decay) not only shrinks weights but also encourages the network to use all its parameters more efficiently.

### Early Stopping

Monitors validation performance and stops training when overfitting begins.

**Implementation Strategy:**
- Track validation loss/metric over epochs
- Stop when validation performance doesn't improve for $`k`$ epochs
- Save best model weights during training

> **Common Pitfall:**
> 
> If your validation set is too small or not representative, early stopping may trigger too soon or too late.

### Data Augmentation

Expands training data through transformations to improve generalization.

**Common Techniques:**
- **Image Data:** Rotation, scaling, flipping, color jittering, cropping
- **Text Data:** Synonym replacement, back-translation, random insertion/deletion
- **Audio Data:** Time stretching, pitch shifting, noise injection

> **Try it yourself!**
> 
> Apply data augmentation to a small dataset and observe the effect on model generalization.

---

## Normalization Techniques

Normalization techniques stabilize training by controlling the distribution of activations and gradients.

> **Key Insight:**
> 
> Normalization not only speeds up training but can also make deeper architectures feasible by reducing internal covariate shift.

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

> **Common Pitfall:**
> 
> BatchNorm can behave poorly with very small batch sizes. Consider GroupNorm or LayerNorm in those cases.

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

> **Did you know?**
> 
> LayerNorm is especially effective in transformer architectures and recurrent neural networks.

### Instance Normalization

Normalizes each individual sample across spatial dimensions.

**Mathematical Formulation:**
```math
\text{IN}(x) = \gamma \frac{x - \mu_I}{\sqrt{\sigma_I^2 + \epsilon}} + \beta
```

Where:
- $`\mu_I = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{h,w}`$ (instance mean)
- $`\sigma_I^2 = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} (x_{h,w} - \mu_I)^2`$ (instance variance)

> **Try it yourself!**
> 
> Use InstanceNorm in a style transfer network and observe the effect on generated images.

### Group Normalization

Normalizes within groups of channels.

**Mathematical Formulation:**
```math
\text{GN}(x) = \gamma \frac{x - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} + \beta
```

Where normalization is performed within each group of $`G`$ channels.

> **Key Insight:**
> 
> GroupNorm is a robust alternative to BatchNorm when batch sizes are small or variable.

---

## Initialization Strategies

Proper weight initialization is crucial for training deep networks effectively.

> **Did you know?**
> 
> The right initialization can make the difference between a model that learns and one that gets stuck with vanishing or exploding gradients.

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

> **Common Pitfall:**
> 
> Using He initialization with sigmoid or tanh can cause activations to explode. Always match initialization to activation!

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

> **Key Insight:**
> 
> Orthogonal initialization is especially powerful for RNNs and very deep networks.

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

> **Try it yourself!**
> 
> Fine-tune a pre-trained model on a small dataset and compare the results to training from scratch.

---

## Practical Considerations

### Choosing Regularization Techniques

| Technique      | Best For                | Key Parameter(s) | Common Pitfall                  |
|---------------|-------------------------|------------------|---------------------------------|
| Dropout       | Fully connected layers  | $`p$` (rate)     | Don't use during inference      |
| Weight Decay  | Most architectures      | $`\lambda$`      | Over-regularization possible    |
| Early Stopping| Any model               | $`k$` (patience) | Needs good validation set       |
| Data Augment. | Images, text, audio     | Transform types  | Must preserve semantics         |

### Normalization Selection

| Technique    | Best For                | Batch Size | Key Insight                                 |
|--------------|-------------------------|------------|---------------------------------------------|
| BatchNorm    | CNNs                    | Large      | Fast, stable training, not for small batches|
| LayerNorm    | RNNs, Transformers      | Any        | Good for sequence models                    |
| InstanceNorm | Style transfer, images  | Any        | Good for image generation                   |
| GroupNorm    | Small/var. batch sizes  | Any        | Robust when batch size is small             |

### Initialization Guidelines

| Strategy         | Use With                | Key Formula / Method                |
|------------------|------------------------|-------------------------------------|
| Xavier/Glorot    | Sigmoid, Tanh          | $`\sqrt{2/(fan_{in}+fan_{out})}`$   |
| He               | ReLU, Leaky ReLU       | $`\sqrt{2/fan_{in}}`$               |
| Orthogonal       | RNNs, Transformers     | QR decomposition                    |
| Pre-trained      | Transfer learning      | Load from pre-trained model         |

---

## Detailed Guides

For comprehensive coverage of each topic with mathematical formulations, Python code examples, and practical implementations, see the following detailed guides:

- **[Regularization Methods](01_regularization_methods.md)** - Complete guide covering dropout variants, weight decay implementation, early stopping strategies, and data augmentation techniques for different data types
- **[Normalization Techniques](02_normalization_techniques.md)** - In-depth exploration of BatchNorm, LayerNorm, InstanceNorm, GroupNorm, and Weight Normalization with performance comparisons
- **[Initialization Strategies](03_initialization_strategies.md)** - Detailed analysis of Xavier/Glorot, He, Orthogonal, and Pre-trained weight initialization with mathematical foundations

Each guide includes:
- Mathematical formulations with LaTeX notation
- Python code examples with PyTorch implementations
- Practical considerations and best practices
- Performance comparisons and trade-offs
- Real-world applications and use cases

---

## Actionable Next Steps

- **Experiment:** Try training a model with and without each technique. Observe the effect on overfitting, convergence, and final accuracy.
- **Visualize:** Plot the distribution of activations, gradients, and weights after applying normalization and initialization.
- **Diagnose:** If your model is not learning, check for issues with initialization, normalization, or regularization first!
- **Connect:** See how these techniques interact in the context of optimization and advanced architectures in later chapters.

> **Key Insight:**
> 
> The best deep learning practitioners are relentless experimenters. Use these techniques as a toolkit—adapt, combine, and test them for your specific problem!

## References

- Srivastava, N., et al. "Dropout: A simple way to prevent neural networks from overfitting"
- Ioffe, S., & Szegedy, C. "Batch normalization: Accelerating deep network training by reducing internal covariate shift"
- Ba, J. L., et al. "Layer normalization"
- Glorot, X., & Bengio, Y. "Understanding the difficulty of training deep feedforward neural networks"
- He, K., et al. "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification" 