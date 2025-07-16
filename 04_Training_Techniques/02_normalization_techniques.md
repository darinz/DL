# Normalization Techniques

> **Key Insight:** Normalization is a cornerstone of modern deep learning. It stabilizes training, accelerates convergence, and enables deeper, more powerful networks.

---

## Table of Contents
1. [Batch Normalization (BatchNorm)](#batch-normalization-batchnorm)
2. [Layer Normalization](#layer-normalization)
3. [Instance Normalization](#instance-normalization)
4. [Group Normalization](#group-normalization)
5. [Weight Normalization](#weight-normalization)
6. [Practical Considerations](#practical-considerations)
7. [Summary](#summary)

---

## Batch Normalization (BatchNorm)

Batch Normalization normalizes layer inputs across the batch dimension, making training more stable and allowing higher learning rates.

### Mathematical Formulation

For a batch of inputs $`x \in \mathbb{R}^{B \times C \times H \times W}`$ (for CNNs) or $`x \in \mathbb{R}^{B \times D}`$ (for fully connected layers):

```math
\text{BN}(x) = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
```

Where:
- $`\mu_B = \frac{1}{B}\sum_{i=1}^{B} x_i`$ (batch mean)
- $`\sigma_B^2 = \frac{1}{B}\sum_{i=1}^{B} (x_i - \mu_B)^2`$ (batch variance)
- $`\gamma, \beta`$ are learnable parameters (scale and shift)
- $`\epsilon`$ is a small constant for numerical stability

### Intuition

- **Key Insight:** By normalizing activations, BatchNorm reduces internal covariate shift, making each layer less sensitive to changes in previous layers.
- Allows for higher learning rates and faster convergence.
- Acts as a regularizer, sometimes reducing the need for dropout.

> **Did you know?**
> BatchNorm can sometimes allow you to remove dropout entirely, especially in large models.

### Training vs Inference

**Training Mode:**
- Uses batch statistics: $`\mu_B, \sigma_B^2`$
- Updates running averages: $`\mu_{\text{running}}, \sigma_{\text{running}}^2`$

**Inference Mode:**
- Uses running averages: $`\mu_{\text{running}}, \sigma_{\text{running}}^2`$

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BatchNorm1d(nn.Module):
    """Custom BatchNorm1d implementation for educational purposes"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Batch statistics
        self.register_buffer('batch_mean', torch.zeros(num_features))
        self.register_buffer('batch_var', torch.ones(num_features))
        
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            self.batch_mean = x.mean(dim=0)
            self.batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var
            
            # Normalize using batch statistics
            x_norm = (x - self.batch_mean) / torch.sqrt(self.batch_var + self.eps)
        else:
            # Normalize using running statistics
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        return self.weight * x_norm + self.bias
```

> **Code Commentary:**
> - BatchNorm maintains running averages of mean and variance for use during inference.
> - Learnable parameters $`\gamma`$ and $`\beta`$ allow the network to undo normalization if needed.

### BatchNorm in Neural Networks

```python
class CNNWithBatchNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x
```

> **Try it yourself!**
> Remove BatchNorm layers from the above network and compare training speed and final accuracy.

### Best Practices

- Place BatchNorm after linear/convolutional layers and before activation functions.
- Use higher learning rates with BatchNorm.
- Batch size should not be too small (ideally $`\geq 16`$).

> **Common Pitfall:**
> Using very small batch sizes can make BatchNorm unstable. Consider LayerNorm or GroupNorm for small-batch or online learning.

---

## Layer Normalization

Layer Normalization normalizes across the features of each sample, rather than across the batch.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times D}`$:

```math
\text{LayerNorm}(x) = \gamma \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta
```

Where:
- $`\mu_L = \frac{1}{D}\sum_{j=1}^{D} x_j`$ (mean over features)
- $`\sigma_L^2 = \frac{1}{D}\sum_{j=1}^{D} (x_j - \mu_L)^2`$ (variance over features)

### Intuition

- **Key Insight:** LayerNorm is independent of batch size and works well for recurrent and transformer models.
- Each sample is normalized independently, making it robust to varying batch statistics.

### Python Implementation

```python
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

> **Did you know?**
> LayerNorm is the default normalization in transformer architectures (e.g., BERT, GPT).

---

## Instance Normalization

Instance Normalization normalizes each sample and channel independently, commonly used in style transfer and generative models.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times C \times H \times W}`$:

```math
\text{InstanceNorm}(x) = \gamma \frac{x - \mu_{IC}}{\sqrt{\sigma_{IC}^2 + \epsilon}} + \beta
```

Where:
- $`\mu_{IC}`$ is the mean over spatial dimensions for each channel and sample
- $`\sigma_{IC}^2`$ is the variance over spatial dimensions for each channel and sample

### Intuition

- **Key Insight:** InstanceNorm removes instance-specific contrast, making it ideal for style transfer.
- Each image (and channel) is normalized independently.

### Python Implementation

```python
class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(InstanceNorm2d, self).__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
```

> **Try it yourself!**
> Use InstanceNorm in a style transfer network and observe the effect on generated images.

---

## Group Normalization

Group Normalization divides channels into groups and normalizes within each group, providing a compromise between BatchNorm and LayerNorm.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times C \times H \times W}`$ and $`G`$ groups:

```math
\text{GroupNorm}(x) = \gamma \frac{x - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} + \beta
```

Where:
- $`\mu_G`$ is the mean over each group
- $`\sigma_G^2`$ is the variance over each group

### Intuition

- **Key Insight:** GroupNorm is robust to small batch sizes and is widely used in object detection and segmentation models.
- By grouping channels, it balances the benefits of BatchNorm and LayerNorm.

### Python Implementation

```python
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        N, C, H, W = x.shape
        G = self.num_groups
        x = x.view(N, G, C // G, H, W)
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(N, C, H, W)
        return self.gamma.view(1, -1, 1, 1) * x_norm + self.beta.view(1, -1, 1, 1)
```

> **Did you know?**
> GroupNorm is the default normalization in many state-of-the-art object detection and segmentation models (e.g., Mask R-CNN).

---

## Weight Normalization

Weight Normalization reparameterizes the weights of a layer to decouple their magnitude from direction, improving optimization.

### Mathematical Formulation

For a weight vector $`w`$:

```math
w = g \frac{v}{\|v\|}
```

Where:
- $`g`$ is a learnable scalar parameter (magnitude)
- $`v`$ is a learnable vector parameter (direction)

### Intuition

- **Key Insight:** WeightNorm speeds up convergence by making the optimization landscape smoother.
- Decouples the scale of weights from their direction.

### Python Implementation

```python
class WeightNormLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WeightNormLinear, self).__init__()
        self.v = nn.Parameter(torch.randn(out_features, in_features))
        self.g = nn.Parameter(torch.ones(out_features, 1))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        w = self.g * self.v / (self.v.norm(dim=1, keepdim=True) + 1e-8)
        return F.linear(x, w, self.bias)
```

> **Try it yourself!**
> Compare training speed and final accuracy with and without WeightNorm on a simple MLP.

---

## Practical Considerations

- Choose normalization based on model architecture and batch size:
  - **BatchNorm:** Best for large batches and CNNs
  - **LayerNorm:** Best for transformers and RNNs
  - **GroupNorm:** Best for small batches and segmentation
  - **InstanceNorm:** Best for style transfer and generative models
- Always use $`\epsilon`$ for numerical stability
- Tune momentum and group size as hyperparameters

> **Common Pitfall:**
> Forgetting to switch normalization layers to evaluation mode (`model.eval()`) can lead to poor inference performance due to incorrect statistics.

---

## Summary

Normalization techniques are essential for stable and efficient deep learning:

- $`\textbf{BatchNorm}`$: Normalizes across the batch, accelerates training
- $`\textbf{LayerNorm}`$: Normalizes across features, robust to batch size
- $`\textbf{InstanceNorm}`$: Normalizes per sample/channel, great for style transfer
- $`\textbf{GroupNorm}`$: Normalizes within groups, robust to small batches
- $`\textbf{WeightNorm}`$: Decouples weight magnitude and direction for smoother optimization

> **Key Insight:**
> The right normalization can make deep networks train faster, generalize better, and reach higher accuracy. Experiment and choose based on your architecture and data! 