# Modern Architectures

Modern CNN architectures focus on efficiency, scalability, and deployment on resource-constrained devices. These architectures introduce novel design principles to achieve better performance with fewer parameters and computational resources.

## Table of Contents

1. [EfficientNet (2019)](#efficientnet-2019)
2. [MobileNet (2017)](#mobilenet-2017)
3. [ShuffleNet (2017)](#shufflenet-2017)
4. [Design Principles](#design-principles)
5. [Performance Comparison](#performance-comparison)

## EfficientNet (2019)

### Historical Context

EfficientNet, developed by Google Research, introduced compound scaling that uniformly scales network depth, width, and resolution using a compound coefficient. It achieved state-of-the-art accuracy with significantly fewer parameters.

### Compound Scaling Method

The key innovation is compound scaling, which scales all three dimensions (depth, width, resolution) together:

```math
\text{depth}: d = \alpha^\phi
\text{width}: w = \beta^\phi  
\text{resolution}: r = \gamma^\phi
\text{where } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
```

Where $`\phi`$ is the compound coefficient that controls resource scaling.

### Mathematical Foundation

**Scaling Equations:**
```math
F_i = \hat{F}_i^{(\alpha \cdot \beta^2 \cdot \gamma^2)^i}
C_i = \hat{C}_i \cdot \beta^i
H_i = \hat{H}_i \cdot \gamma^i
W_i = \hat{W}_i \cdot \gamma^i
```

Where:
- $`F_i`$: Number of layers in stage $`i`$
- $`C_i`$: Number of channels in stage $`i`$
- $`H_i, W_i`$: Input resolution in stage $`i`$

**Computational Cost:**
```math
\text{FLOPs} \propto \alpha \cdot \beta^2 \cdot \gamma^2
```

### Implementation

```python
import torch
import torch.nn as nn
import math

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio=0.25, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, 
                                       kernel_size, stride=stride, 
                                       padding=kernel_size//2, groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(expanded_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.drop_connect_rate = drop_connect_rate
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        
        # Expansion
        x = torch.relu(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise convolution
        x = torch.relu(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze-and-Excitation
        x = x * self.se(x)
        
        # Projection
        x = self.project_bn(self.project_conv(x))
        
        # Residual connection
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate > 0:
                x = self._drop_connect(x)
            x = x + residual
            
        return x
    
    def _drop_connect(self, x):
        if self.training:
            keep_prob = 1 - self.drop_connect_rate
            mask = torch.zeros_like(x).bernoulli_(keep_prob)
            x = x.div(keep_prob) * mask
        return x

class EfficientNet(nn.Module):
    def __init__(self, width_coeff=1.0, depth_coeff=1.0, resolution=224, 
                 dropout_rate=0.2, num_classes=1000):
        super(EfficientNet, self).__init__()
        
        # Base configuration for EfficientNet-B0
        base_config = [
            # t, c, n, s, k
            [1, 16, 1, 1, 3],  # MBConv1, 3x3
            [6, 24, 2, 2, 3],  # MBConv6, 3x3
            [6, 40, 2, 2, 5],  # MBConv6, 5x5
            [6, 80, 3, 2, 3],  # MBConv6, 3x3
            [6, 112, 3, 1, 5], # MBConv6, 5x5
            [6, 192, 4, 2, 5], # MBConv6, 5x5
            [6, 320, 1, 1, 3], # MBConv6, 3x3
        ]
        
        # Scale configuration
        scaled_config = []
        for t, c, n, s, k in base_config:
            scaled_c = int(c * width_coeff)
            scaled_n = int(n * depth_coeff)
            scaled_config.append([t, scaled_c, scaled_n, s, k])
        
        # Initial convolution
        initial_channels = int(32 * width_coeff)
        self.conv1 = nn.Conv2d(3, initial_channels, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        
        # MBConv blocks
        self.blocks = nn.ModuleList()
        in_channels = initial_channels
        
        for t, c, n, s, k in scaled_config:
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(MBConvBlock(in_channels, c, k, stride, t))
                in_channels = c
        
        # Final layers
        final_channels = int(1280 * width_coeff)
        self.conv_head = nn.Conv2d(in_channels, final_channels, 1, bias=False)
        self.bn_head = nn.BatchNorm2d(final_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_channels, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        
        for block in self.blocks:
            x = block(x)
        
        x = torch.relu(self.bn_head(self.conv_head(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

# Create EfficientNet-B0 (base model)
model = EfficientNet(width_coeff=1.0, depth_coeff=1.0, resolution=224)
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

### Key Features

- **Compound scaling**: Uniform scaling of depth, width, and resolution
- **MBConv blocks**: Mobile inverted bottleneck convolution
- **Squeeze-and-Excitation**: Channel attention mechanism
- **Dropout**: Stochastic depth for regularization

## MobileNet (2017)

### Historical Context

MobileNet, developed by Google, introduced depthwise separable convolutions to create efficient neural networks for mobile and embedded vision applications.

### Depthwise Separable Convolution

The key innovation is decomposing standard convolution into two steps:

**1. Depthwise Convolution:**
```math
(I * K_d)(i, j, c) = \sum_{m,n} I(i+m, j+n, c) \cdot K_d(m, n, c)
```

**2. Pointwise Convolution:**
```math
(F * K_p)(i, j, k) = \sum_{c} F(i, j, c) \cdot K_p(c, k)
```

**Computational Reduction:**
```math
\text{Standard Conv}: O(H \times W \times F \times F \times C_{in} \times C_{out})
\text{Depthwise Separable}: O(H \times W \times F \times F \times C_{in} + H \times W \times C_{in} \times C_{out})
```

### Implementation

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                  stride=stride, padding=padding, groups=in_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = torch.relu(self.depthwise_bn(self.depthwise(x)))
        x = torch.relu(self.pointwise_bn(self.pointwise(x)))
        return x

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, width_multiplier=1.0):
        super(MobileNet, self).__init__()
        
        # Configuration: [t, c, n, s]
        # t: expansion factor, c: output channels, n: number of repeats, s: stride
        config = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Initial convolution
        input_channels = int(32 * width_multiplier)
        self.conv1 = nn.Conv2d(3, input_channels, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channels)
        
        # Depthwise separable blocks
        self.blocks = nn.ModuleList()
        
        for t, c, n, s in config:
            output_channels = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(DepthwiseSeparableConv(input_channels, output_channels, 3, stride, 1))
                input_channels = output_channels
        
        # Final layers
        self.conv2 = nn.Conv2d(input_channels, int(1280 * width_multiplier), 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(1280 * width_multiplier))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(int(1280 * width_multiplier), num_classes)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        
        for block in self.blocks:
            x = block(x)
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# Create MobileNet
model = MobileNet(width_multiplier=1.0)
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

### Key Features

- **Depthwise separable convolution**: Reduces computational cost
- **Width multiplier**: Scales network capacity
- **Resolution multiplier**: Scales input resolution
- **Mobile-optimized**: Designed for mobile devices

## ShuffleNet (2017)

### Historical Context

ShuffleNet, developed by Megvii Inc., introduced channel shuffling to enable efficient group convolutions while maintaining accuracy.

### Channel Shuffling

The key innovation is channel shuffling, which enables information flow between groups:

```math
\text{Shuffle}(x) = \text{Reshape}(\text{Transpose}(\text{Reshape}(x, g, c/g, h, w), 1, 2), c, h, w)
```

Where $`g`$ is the number of groups and $`c`$ is the number of channels.

### Implementation

```python
class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = channels // self.groups
        
        # Reshape to (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        
        # Transpose to (batch_size, channels_per_group, groups, height, width)
        x = x.transpose(1, 2).contiguous()
        
        # Reshape back to (batch_size, channels, height, width)
        x = x.view(batch_size, channels, height, width)
        
        return x

class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(ShuffleUnit, self).__init__()
        
        self.stride = stride
        self.groups = groups
        
        # For stride=2, we need to handle dimension mismatch
        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(3, stride=2, padding=1),
                nn.Conv2d(in_channels, out_channels, 1, groups=groups, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Main path
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.shuffle = ChannelShuffle(groups)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, 
                              padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        
        # Main path
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.shuffle(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Residual connection
        if self.stride == 1:
            x = x + residual
        else:
            x = x + self.shortcut(residual)
            
        return torch.relu(x)

class ShuffleNet(nn.Module):
    def __init__(self, num_classes=1000, groups=3):
        super(ShuffleNet, self).__init__()
        
        # Configuration: [out_channels, num_units, stride]
        config = [
            [144, 4, 2],  # Stage 2
            [288, 8, 2],  # Stage 3
            [576, 4, 2],  # Stage 4
        ]
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ShuffleNet stages
        self.stages = nn.ModuleList()
        in_channels = 24
        
        for out_channels, num_units, stride in config:
            stage = nn.Sequential()
            for i in range(num_units):
                unit_stride = stride if i == 0 else 1
                stage.add_module(f'unit_{i}', 
                               ShuffleUnit(in_channels, out_channels, unit_stride, groups))
                in_channels = out_channels
            self.stages.append(stage)
        
        # Final layers
        self.conv5 = nn.Conv2d(in_channels, 1024, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        for stage in self.stages:
            x = stage(x)
        
        x = torch.relu(self.bn5(self.conv5(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x

# Create ShuffleNet
model = ShuffleNet(groups=3)
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

### Key Features

- **Channel shuffling**: Enables efficient group convolutions
- **Group convolutions**: Reduces computational cost
- **Residual connections**: Maintains gradient flow
- **Mobile-friendly**: Optimized for mobile devices

## Design Principles

### 1. Efficiency Metrics

**Computational Efficiency:**
```math
\text{FLOPs} = \sum_{l} H_l \times W_l \times C_{in,l} \times C_{out,l} \times K_l^2
```

**Parameter Efficiency:**
```math
\text{Parameters} = \sum_{l} C_{in,l} \times C_{out,l} \times K_l^2 + C_{out,l}
```

**Memory Efficiency:**
```math
\text{Memory} = \sum_{l} H_l \times W_l \times C_l
```

### 2. Scaling Strategies

**Width Scaling:**
```math
C_{new} = \alpha \times C_{original}
```

**Depth Scaling:**
```math
L_{new} = \beta \times L_{original}
```

**Resolution Scaling:**
```math
H_{new} \times W_{new} = \gamma \times H_{original} \times W_{original}
```

### 3. Architecture Patterns

**Inverted Residual:**
```math
\text{Input} \xrightarrow{\text{Expand}} \text{Depthwise} \xrightarrow{\text{Project}} \text{Output}
```

**Bottleneck Design:**
```math
\text{Input} \xrightarrow{1 \times 1} \text{Conv} \xrightarrow{1 \times 1} \text{Output}
```

## Performance Comparison

### Accuracy vs Efficiency Trade-off

| Architecture | Top-1 Accuracy | Parameters (M) | FLOPs (M) |
|--------------|----------------|----------------|-----------|
| EfficientNet-B0 | 77.1% | 5.3 | 390 |
| MobileNet-v1 | 70.6% | 4.2 | 569 |
| ShuffleNet-v1 | 67.4% | 1.9 | 140 |

### Deployment Considerations

**Mobile Devices:**
- **Latency**: Real-time inference requirements
- **Memory**: Limited RAM constraints
- **Battery**: Power consumption optimization

**Edge Devices:**
- **Model size**: Storage limitations
- **Computational resources**: Limited processing power
- **Network connectivity**: Offline operation capability

## Summary

Modern CNN architectures focus on:

1. **Efficiency**: Reducing computational cost and memory usage
2. **Scalability**: Flexible scaling for different resource constraints
3. **Deployment**: Optimization for mobile and edge devices
4. **Innovation**: Novel architectural patterns (compound scaling, depthwise separable convolution, channel shuffling)

These architectures enable the deployment of powerful neural networks on resource-constrained devices, making AI accessible in mobile and embedded applications. 