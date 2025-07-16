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

> **Explanation:**
> BatchNorm addresses the problem of "internal covariate shift"—the change in the distribution of network activations due to parameter updates during training. By normalizing the inputs to each layer, BatchNorm stabilizes and accelerates training, and can also act as a regularizer.

### Mathematical Formulation

For a batch of inputs $`x \in \mathbb{R}^{B \times C \times H \times W}`$ (for CNNs) or $`x \in \mathbb{R}^{B \times D}`$ (for fully connected layers):

```math
\text{BN}(x) = \gamma \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta
```

> **Math Breakdown:**
> - $x$ is the input tensor (could be activations from a previous layer).
> - $\mu_B$ is the mean of the batch.
> - $\sigma_B^2$ is the variance of the batch.
> - $\epsilon$ is a small constant to prevent division by zero.
> - $\gamma$ and $\beta$ are learnable parameters that allow the network to scale and shift the normalized output.
> - The normalization is performed per feature/channel, not across all elements.

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

> **Explanation:**
> During training, the mean and variance are computed from the current batch. During inference, the running averages (computed during training) are used to ensure consistent behavior.

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

> **Code Walkthrough:**
> - The class maintains running averages of mean and variance for use during inference.
> - During training, normalization uses the current batch's statistics; during inference, it uses the running averages.
> - The learnable parameters $\gamma$ (weight) and $\beta$ (bias) allow the network to undo normalization if needed.

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

> **Code Walkthrough:**
> - BatchNorm layers are placed after convolutional or linear layers and before activation functions.
> - This helps stabilize the distribution of activations throughout the network, improving training speed and performance.

### Best Practices

- Place BatchNorm after linear/convolutional layers and before activation functions.
- Use higher learning rates with BatchNorm.
- Batch size should not be too small (ideally $`\geq 16`$).

> **Common Pitfall:**
> Using very small batch sizes can make BatchNorm unstable. Consider LayerNorm or GroupNorm for small-batch or online learning.

---

## Layer Normalization

Layer Normalization normalizes across the features of each sample, rather than across the batch.

> **Explanation:**
> LayerNorm is especially useful in settings where batch statistics are not reliable, such as in recurrent neural networks (RNNs) or transformers, or when batch sizes are very small. It normalizes each sample independently, making it robust to varying batch statistics.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times D}`$:

```math
\text{LayerNorm}(x) = \gamma \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta
```

> **Math Breakdown:**
> - $x$ is the input vector for a single sample.
> - $\mu_L$ is the mean over the features of that sample.
> - $\sigma_L^2$ is the variance over the features of that sample.
> - $\gamma$ and $\beta$ are learnable parameters for scaling and shifting.
> - $\epsilon$ is a small constant for numerical stability.

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

> **Code Walkthrough:**
> - The normalization is performed across the last dimension (features) for each sample.
> - Learnable parameters allow the network to scale and shift the normalized output as needed.

> **Did you know?**
> LayerNorm is the default normalization in transformer architectures (e.g., BERT, GPT).

---

## Instance Normalization

Instance Normalization normalizes each sample and channel independently, commonly used in style transfer and generative models.

> **Explanation:**
> InstanceNorm is especially useful for tasks like style transfer, where the goal is to normalize the contrast and style of each image independently. By normalizing each sample and channel, it removes instance-specific contrast information, making the model focus on content rather than style.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times C \times H \times W}`$:

```math
\text{InstanceNorm}(x) = \gamma \frac{x - \mu_{IC}}{\sqrt{\sigma_{IC}^2 + \epsilon}} + \beta
```

Where:
- $`\mu_{IC}`$ is the mean over spatial dimensions for each channel and sample
- $`\sigma_{IC}^2`$ is the variance over spatial dimensions for each channel and sample

> **Math Breakdown:**
> - $\mu_{IC}$ and $\sigma_{IC}^2$ are computed for each sample and channel, not across the batch.
> - This makes normalization independent for each image and channel, which is ideal for style transfer.

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

> **Code Walkthrough:**
> - The normalization is performed over the spatial dimensions for each sample and channel.
> - Learnable parameters allow the network to scale and shift the normalized output.

> **Try it yourself!**
> Use InstanceNorm in a style transfer network and observe the effect on generated images.

---

## Group Normalization

Group Normalization divides channels into groups and normalizes within each group, providing a compromise between BatchNorm and LayerNorm.

> **Explanation:**
> GroupNorm is designed to work well even with small batch sizes, making it a good choice for tasks like object detection and segmentation where batch sizes are often limited by memory.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times C \times H \times W}`$ and $`G`$ groups:

```math
\text{GroupNorm}(x) = \gamma \frac{x - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} + \beta
```

Where:
- $`\mu_G`$ is the mean over each group
- $`\sigma_G^2`$ is the variance over each group

> **Math Breakdown:**
> - Channels are divided into $G$ groups.
> - Mean and variance are computed within each group, not across the whole batch or all features.

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

> **Code Walkthrough:**
> - The input is reshaped to group channels, and normalization is performed within each group.
> - This allows normalization to be effective even with very small batch sizes.

> **Did you know?**
> GroupNorm is the default normalization in many state-of-the-art object detection and segmentation models (e.g., Mask R-CNN).

---

## Weight Normalization

Weight Normalization reparameterizes the weights of a layer to decouple their magnitude from direction, improving optimization.

> **Explanation:**
> WeightNorm separates the length (magnitude) and direction of weight vectors, making optimization easier and often speeding up convergence.

### Mathematical Formulation

For a weight vector $`w`$:

```math
w = g \frac{v}{\|v\|}
```

Where:
- $`g`$ is a learnable scalar parameter (magnitude)
- $`v`$ is a learnable vector parameter (direction)

> **Math Breakdown:**
> - $g$ controls the scale of the weights, $v$ controls the direction.
> - This reparameterization can make the optimization landscape smoother.

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

> **Code Walkthrough:**
> - The weight vector is reparameterized into a direction (`v`) and a magnitude (`g`).
> - This can help the optimizer make more effective updates.

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

> **Practical Tip:**
> If your model is unstable or not converging, try switching to a different normalization method or adjusting $\epsilon$.

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