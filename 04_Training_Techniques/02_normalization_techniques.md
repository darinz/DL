# Normalization Techniques

Normalization techniques are crucial for stabilizing the training of deep neural networks by controlling the distribution of activations and gradients. This guide covers the most important normalization methods with detailed explanations, mathematical formulations, and practical Python implementations.

## Table of Contents

1. [Batch Normalization (BatchNorm)](#batch-normalization-batchnorm)
2. [Layer Normalization](#layer-normalization)
3. [Instance Normalization](#instance-normalization)
4. [Group Normalization](#group-normalization)
5. [Weight Normalization](#weight-normalization)
6. [Practical Considerations](#practical-considerations)

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

class BatchNorm2d(nn.Module):
    """Custom BatchNorm2d implementation for CNNs"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        if self.training:
            # Compute batch statistics across spatial dimensions
            batch_mean = x.mean(dim=[0, 2, 3])  # Mean across batch, height, width
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize
            x_norm = (x - batch_mean.view(1, -1, 1, 1)) / torch.sqrt(batch_var.view(1, -1, 1, 1) + self.eps)
        else:
            # Normalize using running statistics
            x_norm = (x - self.running_mean.view(1, -1, 1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + self.eps)
        
        # Scale and shift
        return self.weight.view(1, -1, 1, 1) * x_norm + self.bias.view(1, -1, 1, 1)

# Example usage
def demonstrate_batchnorm():
    # 1D BatchNorm (for fully connected layers)
    batch_size, features = 32, 64
    x_1d = torch.randn(batch_size, features)
    
    bn_1d = BatchNorm1d(features)
    
    # Training mode
    bn_1d.train()
    output_train = bn_1d(x_1d)
    print(f"1D BatchNorm training output - Mean: {output_train.mean():.4f}, Std: {output_train.std():.4f}")
    
    # Evaluation mode
    bn_1d.eval()
    output_eval = bn_1d(x_1d)
    print(f"1D BatchNorm eval output - Mean: {output_eval.mean():.4f}, Std: {output_eval.std():.4f}")
    
    # 2D BatchNorm (for CNNs)
    batch_size, channels, height, width = 16, 32, 28, 28
    x_2d = torch.randn(batch_size, channels, height, width)
    
    bn_2d = BatchNorm2d(channels)
    
    # Training mode
    bn_2d.train()
    output_train_2d = bn_2d(x_2d)
    print(f"2D BatchNorm training output - Mean: {output_train_2d.mean():.4f}, Std: {output_train_2d.std():.4f}")

demonstrate_batchnorm()
```

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

# Training with BatchNorm
def train_with_batchnorm():
    model = CNNWithBatchNorm()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy data
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))
    
    # Training step
    model.train()
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    print(f"Training loss: {loss.item():.4f}")
    
    # Evaluation step
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        print(f"Evaluation outputs shape: {outputs.shape}")

train_with_batchnorm()
```

### BatchNorm Variants

#### 1. BatchNorm with Learnable Parameters

```python
class LearnableBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(LearnableBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Learnable running statistics (unusual but possible)
        self.register_parameter('running_mean', nn.Parameter(torch.zeros(num_features)))
        self.register_parameter('running_var', nn.Parameter(torch.ones(num_features)))
        
    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * batch_mean
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * batch_var
            
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return self.weight * x_norm + self.bias
```

#### 2. BatchNorm with Different Momentum

```python
class AdaptiveBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('step_count', torch.tensor(0))
        
    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Adaptive momentum based on step count
            adaptive_momentum = min(self.momentum, 1.0 / (self.step_count + 1))
            
            self.running_mean = (1 - adaptive_momentum) * self.running_mean + adaptive_momentum * batch_mean
            self.running_var = (1 - adaptive_momentum) * self.running_var + adaptive_momentum * batch_var
            
            self.step_count += 1
            
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return self.weight * x_norm + self.bias
```

## Layer Normalization

Layer Normalization normalizes across the feature dimension for each sample independently, making it independent of batch size.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times D}`$:

```math
\text{LN}(x) = \gamma \frac{x - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} + \beta
```

Where:
- $`\mu_L = \frac{1}{D}\sum_{i=1}^{D} x_i`$ (layer mean)
- $`\sigma_L^2 = \frac{1}{D}\sum_{i=1}^{D} (x_i - \mu_L)^2`$ (layer variance)
- $`D`$ is the feature dimension
- $`\gamma, \beta`$ are learnable parameters

### Python Implementation

```python
class LayerNorm(nn.Module):
    """Custom LayerNorm implementation"""
    
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        
    def forward(self, x):
        # Compute mean and variance across the last len(normalized_shape) dimensions
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return self.weight * x_norm + self.bias

# Example usage
def demonstrate_layernorm():
    # For fully connected layers
    batch_size, features = 32, 64
    x = torch.randn(batch_size, features)
    
    ln = LayerNorm(features)
    output = ln(x)
    
    print(f"LayerNorm output - Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    # For sequences (e.g., in transformers)
    batch_size, seq_len, hidden_dim = 16, 20, 512
    x_seq = torch.randn(batch_size, seq_len, hidden_dim)
    
    ln_seq = LayerNorm(hidden_dim)
    output_seq = ln_seq(x_seq)
    
    print(f"Sequence LayerNorm output - Mean: {output_seq.mean():.4f}, Std: {output_seq.std():.4f}")

demonstrate_layernorm()
```

### LayerNorm in Transformers

```python
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

# Example usage
def demonstrate_transformer_block():
    batch_size, seq_len, hidden_dim = 16, 20, 512
    x = torch.randn(seq_len, batch_size, hidden_dim)  # (seq_len, batch, hidden_dim)
    
    transformer_block = TransformerBlock(hidden_dim, num_heads=8)
    output = transformer_block(x)
    
    print(f"Transformer block output shape: {output.shape}")
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")

demonstrate_transformer_block()
```

## Instance Normalization

Instance Normalization normalizes each individual sample across spatial dimensions, commonly used in style transfer and image generation.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times C \times H \times W}`$:

```math
\text{IN}(x) = \gamma \frac{x - \mu_I}{\sqrt{\sigma_I^2 + \epsilon}} + \beta
```

Where:
- $`\mu_I = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} x_{h,w}`$ (instance mean)
- $`\sigma_I^2 = \frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W} (x_{h,w} - \mu_I)^2`$ (instance variance)
- Normalization is performed independently for each sample and channel

### Python Implementation

```python
class InstanceNorm2d(nn.Module):
    """Custom InstanceNorm2d implementation"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(InstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size, num_channels, height, width = x.shape
        
        # Reshape for easier computation
        x_reshaped = x.view(batch_size * num_channels, height * width)
        
        # Compute mean and variance for each instance
        mean = x_reshaped.mean(dim=1, keepdim=True)
        var = x_reshaped.var(dim=1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.view(batch_size, num_channels, height, width)
        
        # Apply affine transformation if enabled
        if self.affine:
            x_norm = self.weight.view(1, -1, 1, 1) * x_norm + self.bias.view(1, -1, 1, 1)
        
        return x_norm

# Example usage
def demonstrate_instancenorm():
    batch_size, channels, height, width = 4, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)
    
    in_norm = InstanceNorm2d(channels)
    output = in_norm(x)
    
    print(f"InstanceNorm output shape: {output.shape}")
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")
    
    # Check that each instance is normalized
    for i in range(batch_size):
        for j in range(channels):
            instance_mean = output[i, j].mean()
            instance_std = output[i, j].std()
            print(f"Instance {i}, Channel {j}: Mean={instance_mean:.4f}, Std={instance_std:.4f}")

demonstrate_instancenorm()
```

### InstanceNorm in Style Transfer

```python
class StyleTransferBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StyleTransferBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = self.relu(x)
        return x

class StyleTransferNetwork(nn.Module):
    def __init__(self):
        super(StyleTransferNetwork, self).__init__()
        self.encoder = nn.Sequential(
            StyleTransferBlock(3, 64),
            StyleTransferBlock(64, 128),
            StyleTransferBlock(128, 256)
        )
        
        self.decoder = nn.Sequential(
            StyleTransferBlock(256, 128),
            StyleTransferBlock(128, 64),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example usage
def demonstrate_style_transfer():
    batch_size, channels, height, width = 1, 3, 256, 256
    content_image = torch.randn(batch_size, channels, height, width)
    
    style_network = StyleTransferNetwork()
    stylized_image = style_network(content_image)
    
    print(f"Input image shape: {content_image.shape}")
    print(f"Stylized image shape: {stylized_image.shape}")
    print(f"Stylized image range: [{stylized_image.min():.4f}, {stylized_image.max():.4f}]")

demonstrate_style_transfer()
```

## Group Normalization

Group Normalization normalizes within groups of channels, providing a middle ground between BatchNorm and LayerNorm.

### Mathematical Formulation

For input $`x \in \mathbb{R}^{B \times C \times H \times W}`$ with $`G`$ groups:

```math
\text{GN}(x) = \gamma \frac{x - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} + \beta
```

Where normalization is performed within each group of $`C/G`$ channels.

### Python Implementation

```python
class GroupNorm(nn.Module):
    """Custom GroupNorm implementation"""
    
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        assert num_channels % num_groups == 0, 'num_channels must be divisible by num_groups'
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        batch_size, num_channels, height, width = x.shape
        
        # Reshape for group-wise normalization
        x_reshaped = x.view(batch_size, self.num_groups, num_channels // self.num_groups, height, width)
        
        # Compute mean and variance for each group
        mean = x_reshaped.mean(dim=[2, 3, 4], keepdim=True)
        var = x_reshaped.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.view(batch_size, num_channels, height, width)
        
        # Apply affine transformation if enabled
        if self.affine:
            x_norm = self.weight.view(1, -1, 1, 1) * x_norm + self.bias.view(1, -1, 1, 1)
        
        return x_norm

# Example usage
def demonstrate_groupnorm():
    batch_size, channels, height, width = 8, 32, 28, 28
    x = torch.randn(batch_size, channels, height, width)
    
    # GroupNorm with 8 groups (4 channels per group)
    gn = GroupNorm(num_groups=8, num_channels=channels)
    output = gn(x)
    
    print(f"GroupNorm output shape: {output.shape}")
    print(f"Output mean: {output.mean():.4f}, std: {output.std():.4f}")
    
    # Compare with PyTorch implementation
    gn_pytorch = nn.GroupNorm(num_groups=8, num_channels=channels)
    output_pytorch = gn_pytorch(x)
    
    print(f"PyTorch GroupNorm output mean: {output_pytorch.mean():.4f}, std: {output_pytorch.std():.4f}")
    print(f"Difference: {torch.abs(output - output_pytorch).max():.6f}")

demonstrate_groupnorm()
```

### GroupNorm in Different Architectures

```python
class CNNWithGroupNorm(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNWithGroupNorm, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)  # 8 groups, 32 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)  # 8 groups, 64 channels
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))
        x = self.pool(F.relu(self.gn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
def demonstrate_cnn_with_groupnorm():
    model = CNNWithGroupNorm()
    
    # Test with different batch sizes
    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        print(f"Batch size {batch_size}: Output shape {output.shape}")

demonstrate_cnn_with_groupnorm()
```

## Weight Normalization

Weight Normalization reparameterizes the weight vectors to decouple their length from their direction.

### Mathematical Formulation

For a weight vector $`w`$, weight normalization reparameterizes it as:

```math
w = \frac{g}{\|v\|} v
```

Where:
- $`v`$ is the direction vector
- $`g`$ is the scale parameter
- $`\|v\|`$ is the L2 norm of $`v`$

### Python Implementation

```python
class WeightNormLinear(nn.Module):
    """Linear layer with weight normalization"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(WeightNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize direction vector v
        self.v = nn.Parameter(torch.randn(out_features, in_features))
        
        # Initialize scale parameter g
        self.g = nn.Parameter(torch.ones(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.v)
        self.v.data.copy_(self.v.data / self.v.data.norm(dim=1, keepdim=True))
        
    def forward(self, x):
        # Compute normalized weights
        v_norm = self.v / self.v.norm(dim=1, keepdim=True)
        w = self.g.view(-1, 1) * v_norm
        
        # Linear transformation
        return F.linear(x, w, self.bias)

# Example usage
def demonstrate_weightnorm():
    batch_size, in_features, out_features = 32, 784, 256
    x = torch.randn(batch_size, in_features)
    
    wn_linear = WeightNormLinear(in_features, out_features)
    output = wn_linear(x)
    
    print(f"WeightNorm output shape: {output.shape}")
    
    # Check weight norms
    v_norm = wn_linear.v / wn_linear.v.norm(dim=1, keepdim=True)
    w = wn_linear.g.view(-1, 1) * v_norm
    weight_norms = w.norm(dim=1)
    print(f"Weight norms - Min: {weight_norms.min():.4f}, Max: {weight_norms.max():.4f}")

demonstrate_weightnorm()
```

## Practical Considerations

### When to Use Each Normalization Technique

```python
def choose_normalization_technique():
    """Guidelines for choosing normalization techniques"""
    
    guidelines = {
        'BatchNorm': {
            'use_when': [
                'Large batch sizes (>16)',
                'CNNs with sufficient data',
                'Stable training is priority'
            ],
            'avoid_when': [
                'Small batch sizes',
                'Recurrent networks',
                'Variable batch sizes'
            ]
        },
        'LayerNorm': {
            'use_when': [
                'Recurrent networks (RNNs, LSTMs)',
                'Transformers',
                'Variable batch sizes',
                'Sequence data'
            ],
            'avoid_when': [
                'CNNs (use BatchNorm instead)',
                'Small feature dimensions'
            ]
        },
        'InstanceNorm': {
            'use_when': [
                'Style transfer',
                'Image generation',
                'Per-sample normalization needed'
            ],
            'avoid_when': [
                'Classification tasks',
                'When batch statistics are important'
            ]
        },
        'GroupNorm': {
            'use_when': [
                'Small batch sizes',
                'CNNs with limited data',
                'Alternative to BatchNorm'
            ],
            'avoid_when': [
                'Large batch sizes (BatchNorm is better)',
                'Very small number of channels'
            ]
        }
    }
    
    return guidelines

# Example usage
def demonstrate_normalization_choice():
    guidelines = choose_normalization_technique()
    
    for technique, info in guidelines.items():
        print(f"\n{technique}:")
        print("  Use when:")
        for condition in info['use_when']:
            print(f"    - {condition}")
        print("  Avoid when:")
        for condition in info['avoid_when']:
            print(f"    - {condition}")

demonstrate_normalization_choice()
```

### Performance Comparison

```python
def compare_normalization_performance():
    """Compare different normalization techniques"""
    
    batch_size, channels, height, width = 16, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    # Create different normalization layers
    bn = nn.BatchNorm2d(channels)
    ln = nn.LayerNorm([channels, height, width])
    in_norm = nn.InstanceNorm2d(channels)
    gn = nn.GroupNorm(8, channels)  # 8 groups
    
    # Test forward pass time
    import time
    
    normalizations = {
        'BatchNorm': bn,
        'LayerNorm': ln,
        'InstanceNorm': in_norm,
        'GroupNorm': gn
    }
    
    for name, norm in normalizations.items():
        # Warm up
        for _ in range(10):
            _ = norm(x)
        
        # Time forward pass
        start_time = time.time()
        for _ in range(100):
            output = norm(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        print(f"{name}: {avg_time*1000:.2f} ms per forward pass")

compare_normalization_performance()
```

### Memory Usage Comparison

```python
def compare_memory_usage():
    """Compare memory usage of different normalization techniques"""
    
    batch_size, channels, height, width = 16, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)
    
    normalizations = {
        'BatchNorm': nn.BatchNorm2d(channels),
        'LayerNorm': nn.LayerNorm([channels, height, width]),
        'InstanceNorm': nn.InstanceNorm2d(channels),
        'GroupNorm': nn.GroupNorm(8, channels)
    }
    
    for name, norm in normalizations.items():
        # Count parameters
        total_params = sum(p.numel() for p in norm.parameters())
        print(f"{name}: {total_params} parameters")
        
        # Count buffers (for running statistics)
        total_buffers = sum(b.numel() for b in norm.buffers())
        if total_buffers > 0:
            print(f"  Buffers: {total_buffers} elements")

compare_memory_usage()
```

## Summary

Normalization techniques are essential for training deep neural networks effectively:

1. **BatchNorm**: Best for CNNs with large batch sizes, uses batch statistics
2. **LayerNorm**: Ideal for RNNs and transformers, normalizes across features
3. **InstanceNorm**: Perfect for style transfer, normalizes each sample independently
4. **GroupNorm**: Good alternative to BatchNorm for small batch sizes
5. **WeightNorm**: Reparameterizes weights to separate direction and magnitude

The choice of normalization technique depends on:
- **Architecture**: CNN vs RNN vs Transformer
- **Batch size**: Large vs small
- **Task**: Classification vs generation vs style transfer
- **Data characteristics**: Image vs text vs audio

Understanding these techniques and their trade-offs is crucial for building robust deep learning models. 