# Architecture Evolution

The evolution of Convolutional Neural Network architectures has been driven by the need to solve increasingly complex computer vision tasks while addressing challenges like vanishing gradients, overfitting, and computational efficiency.

> **Key Insight:**
> 
> Each new CNN architecture was a response to a specific challenge—deeper networks, better generalization, or more efficient computation. Understanding this evolution helps you design better models for your own tasks.

## Table of Contents

1. [LeNet-5 (1998)](#lenet-5-1998)
2. [AlexNet (2012)](#alexnet-2012)
3. [VGG (2014)](#vgg-2014)
4. [ResNet (2015)](#resnet-2015)
5. [DenseNet (2017)](#densenet-2017)
6. [Key Innovations](#key-innovations)
7. [Summary Table](#summary-table)
8. [Actionable Next Steps](#actionable-next-steps)

---

## LeNet-5 (1998)

### Historical Context

LeNet-5, developed by Yann LeCun and colleagues, was the first successful CNN for digit recognition. It demonstrated the potential of convolutional networks for pattern recognition tasks.

> **Did you know?**
> 
> LeNet-5 was used to read millions of handwritten checks in the 1990s!

### Architecture Overview

```
Input (32x32) → Conv1 (6@28x28) → Pool1 (6@14x14) → Conv2 (16@10x10) → 
Pool2 (16@5x5) → Conv3 (120@1x1) → FC1 (84) → FC2 (10) → Output
```

### Mathematical Formulation

**Convolutional Layer:**
```math
y_{i,j,k} = \sum_{c=1}^{C_{in}} \sum_{m=0}^{F-1} \sum_{n=0}^{F-1} x_{i+m,j+n,c} \cdot w_{m,n,c,k} + b_k
```

**Pooling Layer:**
```math
y_{i,j,k} = \max_{(m,n) \in R_{i,j}} x_{m,n,k}
```

### Implementation

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
        # Activation function (Tanh was used in original LeNet)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        # Convolutional block 1
        x = self.pool(self.activation(self.conv1(x)))  # 32x32 → 28x28 → 14x14
        
        # Convolutional block 2
        x = self.pool(self.activation(self.conv2(x)))  # 14x14 → 10x10 → 5x5
        
        # Convolutional block 3
        x = self.activation(self.conv3(x))  # 5x5 → 1x1
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Create model
model = LeNet5()
print(model)

# Test with sample input
sample_input = torch.randn(1, 1, 32, 32)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

### Key Features

- **7 layers**: 2 convolutional + 2 pooling + 3 fully connected
- **Parameter sharing**: Convolutional layers share weights
- **Subsampling**: Pooling layers reduce spatial dimensions
- **Non-linear activation**: Tanh activation functions

> **Key Insight:**
> 
> LeNet-5 introduced the idea of local receptive fields, weight sharing, and subsampling—concepts that are still fundamental in modern CNNs.

---

## AlexNet (2012)

### Historical Context

AlexNet, developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, won the ImageNet 2012 competition and marked the beginning of the deep learning revolution in computer vision.

> **Did you know?**
> 
> AlexNet was trained on two GPUs in parallel—a major innovation at the time!

### Architecture Overview

```
Input (227x227x3) → Conv1 (96@55x55) → Pool1 (96@27x27) → Conv2 (256@27x27) → 
Pool2 (256@13x13) → Conv3 (384@13x13) → Conv4 (384@13x13) → Conv5 (256@13x13) → 
Pool3 (256@6x6) → FC1 (4096) → FC2 (4096) → FC3 (1000) → Output
```

### Mathematical Innovations

**ReLU Activation:**
```math
f(x) = \max(0, x)
```

**Dropout Regularization:**
```math
y_i = \begin{cases}
\frac{x_i}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}
```

### Implementation

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Conv1: 227x227x3 → 55x55x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 27x27x96 → 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 13x13x256 → 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: 13x13x384 → 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: 13x13x384 → 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create model
model = AlexNet()
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 227, 227)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

### Key Innovations

- **ReLU activation**: Faster training and better gradient flow
- **Dropout**: Prevents overfitting
- **Data augmentation**: Random crops, horizontal flips
- **GPU implementation**: Parallel processing
- **Local response normalization**: Normalizes responses across channels

> **Key Insight:**
> 
> AlexNet showed that deeper networks, trained with the right tricks, could outperform all previous computer vision methods.

---

## VGG (2014)

### Historical Context

VGG, developed by the Visual Geometry Group at Oxford, demonstrated that depth is crucial for performance. It introduced a simple, consistent architecture pattern.

> **Try it yourself!**
> 
> Compare the number of parameters in a VGG-16 model to LeNet-5. How does depth affect capacity and performance?

### Architecture Overview

VGG comes in different variants (VGG-11, VGG-13, VGG-16, VGG-19) with increasing depth:

```
Input (224x224x3) → Conv1 (64) → Conv2 (64) → Pool1 → Conv3 (128) → Conv4 (128) → 
Pool2 → Conv5 (256) → Conv6 (256) → Conv7 (256) → Pool3 → Conv8 (512) → 
Conv9 (512) → Conv10 (512) → Pool4 → Conv11 (512) → Conv12 (512) → 
Conv13 (512) → Pool5 → FC1 (4096) → FC2 (4096) → FC3 (1000) → Output
```

### Mathematical Design Principles

**3x3 Convolution Stacking:**
```math
\text{Receptive Field} = 1 + 2 \times \text{number of 3x3 convolutions}
```

**Parameter Efficiency:**
```math
\text{Parameters for 7x7} = 49 \times C_{in} \times C_{out}
\text{Parameters for 3x3\times3} = 27 \times C_{in} \times C_{out}
```

### Implementation

```python
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_conv):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.conv_block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        return x

class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        # VGG blocks
        self.features = nn.Sequential(
            VGGBlock(3, 64, 2),      # 224x224 → 112x112
            VGGBlock(64, 128, 2),    # 112x112 → 56x56
            VGGBlock(128, 256, 3),   # 56x56 → 28x28
            VGGBlock(256, 512, 3),   # 28x28 → 14x14
            VGGBlock(512, 512, 3),   # 14x14 → 7x7
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create model
model = VGG16()
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

### Key Features

- **Consistent design**: 3x3 convolutions with 2x2 pooling
- **Increasing depth**: More layers for better feature learning
- **Simple architecture**: Easy to understand and implement
- **Transfer learning**: Excellent pre-trained models

> **Key Insight:**
> 
> VGG's simplicity and depth made it a favorite for transfer learning and feature extraction in many applications.

---

## ResNet (2015)

### Historical Context

ResNet, developed by Microsoft Research, solved the vanishing gradient problem in very deep networks by introducing residual connections. It won the ImageNet 2015 competition.

> **Did you know?**
> 
> ResNet-152 has 152 layers, but can be trained as easily as a 20-layer network thanks to residual connections!

### Architecture Overview

ResNet comes in different variants (ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152):

```
Input → Conv1 → MaxPool → ResBlock1 → ResBlock2 → ResBlock3 → ResBlock4 → 
GlobalAvgPool → FC → Output
```

### Mathematical Innovation

**Residual Connection:**
```math
F(x) = H(x) - x
y = H(x) = F(x) + x
```

**Residual Block:**
```math
y_l = h(x_l) + F(x_l, W_l)
x_{l+1} = f(y_l)
```

Where $`h(x_l) = x_l`$ for identity mapping.

### Implementation

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        
        # Main path
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add shortcut
        out += self.shortcut(residual)
        out = torch.relu(out)
        
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(torch.relu(self.bn1(self.conv1(x))))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Create model
model = ResNet18()
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

### Key Innovations

- **Residual connections**: Solve vanishing gradient problem
- **Batch normalization**: Stabilize training
- **Identity mapping**: Preserve information flow
- **Deep networks**: Enable training of 100+ layer networks

> **Key Insight:**
> 
> Residual connections allow gradients to flow directly through the network, making very deep architectures practical.

---

## DenseNet (2017)

### Historical Context

DenseNet, developed by Facebook AI Research, introduced dense connections where each layer connects to all subsequent layers, promoting feature reuse and gradient flow.

> **Try it yourself!**
> 
> Visualize the feature maps in a DenseNet. How does feature reuse affect the learned representations?

### Architecture Overview

```
Input → Conv1 → DenseBlock1 → Transition1 → DenseBlock2 → Transition2 → 
DenseBlock3 → Transition3 → DenseBlock4 → GlobalAvgPool → FC → Output
```

### Mathematical Innovation

**Dense Connection:**
```math
x_l = H_l([x_0, x_1, ..., x_{l-1}])
```

Where $`[x_0, x_1, ..., x_{l-1}]`$ denotes concatenation of feature maps.

**Growth Rate:**
```math
\text{Number of channels in layer } l = k_0 + k \times l
```

Where $`k`$ is the growth rate and $`k_0`$ is the number of input channels.

### Implementation

```python
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, 
                              padding=1, bias=False)
        
    def forward(self, x):
        out = torch.relu(self.bn1(x))
        out = self.conv1(out)
        out = torch.relu(self.bn2(out))
        out = self.conv2(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = torch.relu(self.bn(x))
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000):
        super(DenseNet, self).__init__()
        
        # Initial convolution
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks
        self.denseblock1 = DenseBlock(num_channels, growth_rate, block_config[0])
        num_channels += block_config[0] * growth_rate
        self.transition1 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.denseblock2 = DenseBlock(num_channels, growth_rate, block_config[1])
        num_channels += block_config[1] * growth_rate
        self.transition2 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.denseblock3 = DenseBlock(num_channels, growth_rate, block_config[2])
        num_channels += block_config[2] * growth_rate
        self.transition3 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        self.denseblock4 = DenseBlock(num_channels, growth_rate, block_config[3])
        num_channels += block_config[3] * growth_rate
        
        # Final layers
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_channels, num_classes)
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        x = self.denseblock1(x)
        x = self.transition1(x)
        
        x = self.denseblock2(x)
        x = self.transition2(x)
        
        x = self.denseblock3(x)
        x = self.transition3(x)
        
        x = self.denseblock4(x)
        
        x = torch.relu(self.bn_final(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Create model
model = DenseNet()
print(model)

# Test with sample input
sample_input = torch.randn(1, 3, 224, 224)
output = model(sample_input)
print(f"Input shape: {sample_input.shape}")
print(f"Output shape: {output.shape}")
```

### Key Innovations

- **Dense connections**: Each layer connects to all subsequent layers
- **Feature reuse**: Efficient use of features
- **Gradient flow**: Better gradient propagation
- **Parameter efficiency**: Fewer parameters than traditional CNNs

> **Key Insight:**
> 
> DenseNet's dense connectivity pattern encourages feature reuse and makes the network more parameter-efficient.

---

## Key Innovations

### 1. Activation Functions

**Evolution:**
- **Sigmoid/Tanh** (LeNet): Suffer from vanishing gradients
- **ReLU** (AlexNet): Faster training, better gradient flow
- **Leaky ReLU**: Prevents dying ReLU problem
- **Swish/GELU**: Smooth, non-monotonic activations

### 2. Regularization Techniques

**Dropout:**
```math
y_i = \begin{cases}
\frac{x_i}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}
```

**Batch Normalization:**
```math
y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

### 3. Optimization Strategies

**Learning Rate Scheduling:**
```math
\text{LR}(t) = \text{LR}_0 \times \left(\frac{1}{2}\right)^{\lfloor t/T \rfloor}
```

**Weight Decay:**
```math
L_{reg} = L + \lambda \sum_{w} w^2
```

### 4. Architecture Patterns

**Inception Module:**
```math
y = \text{Concat}[\text{Branch}_1(x), \text{Branch}_2(x), \text{Branch}_3(x), \text{Branch}_4(x)]
```

**Squeeze-and-Excitation:**
```math
s = \sigma(W_2 \text{ReLU}(W_1 \text{GAP}(x)))
y = s \odot x
```

---

## Summary Table

| Architecture | Year | Key Innovation(s)                | Depth | Activation | Regularization | Notable Use Case         |
|--------------|------|----------------------------------|-------|------------|----------------|-------------------------|
| LeNet-5      | 1998 | Local receptive fields, pooling  | 7     | Tanh       | -              | Digit recognition       |
| AlexNet      | 2012 | ReLU, dropout, GPU, data aug.    | 8     | ReLU       | Dropout        | ImageNet, general CV    |
| VGG          | 2014 | Deep, simple, 3x3 convs          | 16-19 | ReLU       | Dropout        | Transfer learning       |
| ResNet       | 2015 | Residual connections             | 18-152| ReLU       | BatchNorm      | Very deep networks      |
| DenseNet     | 2017 | Dense connections, feature reuse | 121+  | ReLU       | BatchNorm      | Efficient deep models   |

---

## Actionable Next Steps

- **Experiment:** Try training a small LeNet, VGG, ResNet, and DenseNet on the same dataset. Compare accuracy, training speed, and parameter count.
- **Visualize:** Plot feature maps and activation distributions for each architecture.
- **Diagnose:** If your deep network is not learning, try adding residual or dense connections.
- **Connect:** See how these architectures influence modern models like EfficientNet, MobileNet, and Vision Transformers in later chapters.

> **Key Insight:**
> 
> The story of CNN architecture evolution is a story of creative problem-solving. Each innovation builds on the last—so keep experimenting and building on what you learn! 