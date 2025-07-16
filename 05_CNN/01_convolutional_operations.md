# Convolutional Operations

Convolutional operations are the fundamental building blocks of Convolutional Neural Networks (CNNs). They perform feature extraction by applying learnable filters (kernels) to input data through a sliding window mechanism.

## Table of Contents

1. [Basic Convolution](#basic-convolution)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Key Properties](#key-properties)
4. [Stride and Padding](#stride-and-padding)
5. [Multi-Channel Convolution](#multi-channel-convolution)
6. [Implementation Examples](#implementation-examples)

## Basic Convolution

### Definition

The discrete convolution operation combines two functions to produce a third function that expresses how the shape of one is modified by the other. In the context of CNNs, we convolve an input feature map with a learnable kernel.

### Mathematical Formulation

For a 2D input $`I`$ and kernel $`K`$, the convolution operation is defined as:

```math
(I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) \cdot K(m, n)
```

Where:
- $`I`$ is the input feature map of size $`H \times W`$
- $`K`$ is the convolutional kernel of size $`F \times F`$
- $`(i, j)`$ are the output coordinates
- $`(m, n)`$ are the kernel coordinates

### Visual Understanding

```
Input:          Kernel:         Output:
┌─────────┐     ┌─────┐        ┌─────┐
│ 1 2 3 4 │     │ 1 0 │        │ 8 9 │
│ 5 6 7 8 │  *  │ 0 1 │   =    │12 13│
│ 9 10 11 12│   └─────┘        └─────┘
└─────────┘
```

## Mathematical Foundation

### Cross-Correlation vs Convolution

In deep learning, we often use cross-correlation instead of true convolution:

**Cross-correlation:**
```math
(I \star K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) \cdot K(m, n)
```

**True convolution:**
```math
(I * K)(i, j) = \sum_{m} \sum_{n} I(i - m, j - n) \cdot K(m, n)
```

The difference is in the sign of the indices. For symmetric kernels, both operations produce the same result.

### Convolution as Matrix Multiplication

Convolution can be expressed as matrix multiplication by flattening the input and using a Toeplitz matrix:

```math
y = \text{vec}(I * K) = C \cdot \text{vec}(I)
```

Where $`C`$ is the convolution matrix constructed from the kernel $`K`$.

## Key Properties

### 1. Local Connectivity

Each output neuron only connects to a local region of the input, reducing the number of parameters and computational complexity.

**Parameter reduction:**
- Fully connected layer: $`O(H \times W \times H' \times W')`$ parameters
- Convolutional layer: $`O(F \times F)`$ parameters per filter

### 2. Weight Sharing

The same filter is applied across the entire input, enabling:
- **Translation invariance**: The network learns to detect features regardless of their position
- **Parameter efficiency**: Significantly fewer parameters than fully connected layers
- **Regularization**: Reduces overfitting through parameter sharing

### 3. Feature Maps

Multiple filters create different feature maps, each detecting specific patterns:
- Edge detectors
- Texture extractors
- Shape recognizers

## Stride and Padding

### Stride

Stride $`s`$ controls the step size of the filter. The output size is calculated as:

```math
H_{out} = \left\lfloor \frac{H_{in} - F + 2P}{s} \right\rfloor + 1
W_{out} = \left\lfloor \frac{W_{in} - F + 2P}{s} \right\rfloor + 1
```

Where:
- $`H_{in}, W_{in}`$: Input height and width
- $`F`$: Filter size
- $`P`$: Padding size
- $`s`$: Stride

### Padding

Padding adds zeros around the input to control output size:

**Valid padding (no padding):**
```math
H_{out} = H_{in} - F + 1
W_{out} = W_{in} - F + 1
```

**Same padding:**
```math
P = \frac{F - 1}{2}
H_{out} = H_{in}
W_{out} = W_{in}
```

## Multi-Channel Convolution

### Single Input Channel

For grayscale images with one channel:

```math
(I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) \cdot K(m, n)
```

### Multiple Input Channels

For RGB images with $`C_{in}`$ input channels:

```math
(I * K)(i, j, k) = \sum_{c=1}^{C_{in}} \sum_{m} \sum_{n} I(i + m, j + n, c) \cdot K(m, n, c, k)
```

Where $`k`$ indexes the output channels.

### Multiple Filters

With $`C_{out}`$ filters, the complete operation is:

```math
\text{Output}(i, j, k) = \sum_{c=1}^{C_{in}} \sum_{m} \sum_{n} \text{Input}(i + m, j + n, c) \cdot \text{Filter}(m, n, c, k)
```

## Implementation Examples

### Basic Convolution Implementation

```python
import numpy as np

def conv2d(input_data, kernel, stride=1, padding=0):
    """
    Basic 2D convolution implementation
    
    Args:
        input_data: Input array of shape (H, W) or (C, H, W)
        kernel: Convolutional kernel of shape (F, F) or (C, F, F)
        stride: Stride of the convolution
        padding: Padding size
    
    Returns:
        Output array after convolution
    """
    # Add padding
    if padding > 0:
        input_data = np.pad(input_data, padding, mode='constant')
    
    # Get dimensions
    H, W = input_data.shape[-2:]
    F = kernel.shape[-1]
    
    # Calculate output dimensions
    H_out = (H - F) // stride + 1
    W_out = (W - F) // stride + 1
    
    # Initialize output
    output = np.zeros((H_out, W_out))
    
    # Perform convolution
    for i in range(0, H_out):
        for j in range(0, W_out):
            h_start = i * stride
            h_end = h_start + F
            w_start = j * stride
            w_end = w_start + F
            
            # Extract the region
            region = input_data[h_start:h_end, w_start:w_end]
            
            # Apply kernel
            output[i, j] = np.sum(region * kernel)
    
    return output

# Example usage
input_data = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])

kernel = np.array([[1, 0],
                   [0, 1]])

result = conv2d(input_data, kernel, stride=1, padding=0)
print("Convolution result:")
print(result)
```

### Multi-Channel Convolution

```python
def conv2d_multi_channel(input_data, kernel, stride=1, padding=0):
    """
    Multi-channel 2D convolution
    
    Args:
        input_data: Input array of shape (C_in, H, W)
        kernel: Convolutional kernel of shape (C_out, C_in, F, F)
        stride: Stride of the convolution
        padding: Padding size
    
    Returns:
        Output array of shape (C_out, H_out, W_out)
    """
    C_in, H, W = input_data.shape
    C_out, _, F, _ = kernel.shape
    
    # Add padding
    if padding > 0:
        input_data = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), 
                           mode='constant')
    
    # Calculate output dimensions
    H_out = (H + 2*padding - F) // stride + 1
    W_out = (W + 2*padding - F) // stride + 1
    
    # Initialize output
    output = np.zeros((C_out, H_out, W_out))
    
    # Perform convolution for each output channel
    for k in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * stride
                h_end = h_start + F
                w_start = j * stride
                w_end = w_start + F
                
                # Extract the region
                region = input_data[:, h_start:h_end, w_start:w_end]
                
                # Apply kernel for this output channel
                output[k, i, j] = np.sum(region * kernel[k])
    
    return output

# Example with RGB image (3 channels)
input_rgb = np.random.rand(3, 32, 32)  # 3 channels, 32x32 image
kernel_rgb = np.random.rand(16, 3, 3, 3)  # 16 output channels, 3 input channels, 3x3 kernel

result_rgb = conv2d_multi_channel(input_rgb, kernel_rgb, stride=1, padding=1)
print(f"Input shape: {input_rgb.shape}")
print(f"Kernel shape: {kernel_rgb.shape}")
print(f"Output shape: {result_rgb.shape}")
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn

# Create a simple CNN layer
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                      stride=1, padding=1)

# Create input tensor (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 32, 32)

# Apply convolution
output = conv_layer(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")

# Access kernel weights
print(f"Kernel shape: {conv_layer.weight.shape}")
print(f"Bias shape: {conv_layer.bias.shape}")
```

### TensorFlow Implementation

```python
import tensorflow as tf

# Create a simple CNN layer
conv_layer = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding='same',
    activation='relu'
)

# Create input tensor
input_tensor = tf.random.normal((1, 32, 32, 3))  # NHWC format

# Apply convolution
output = conv_layer(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
```

## Advanced Concepts

### Dilated Convolution

Dilated convolution (atrous convolution) introduces gaps between kernel elements:

```math
(I * K)(i, j) = \sum_{m} \sum_{n} I(i + r \cdot m, j + r \cdot n) \cdot K(m, n)
```

Where $`r`$ is the dilation rate.

```python
# PyTorch dilated convolution
dilated_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                        dilation=2, padding=2)
```

### Grouped Convolution

Grouped convolution divides input channels into groups and applies separate convolutions:

```python
# PyTorch grouped convolution
grouped_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, 
                        groups=4)  # 4 groups, 8 channels per group
```

### Depthwise Separable Convolution

Depthwise separable convolution consists of two steps:

1. **Depthwise convolution**: Each input channel is convolved separately
2. **Pointwise convolution**: 1x1 convolution to combine channels

```python
# PyTorch depthwise separable convolution
depthwise = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, 
                     groups=32, padding=1)
pointwise = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)

def depthwise_separable_conv(x):
    x = depthwise(x)
    x = pointwise(x)
    return x
```

## Performance Considerations

### Computational Complexity

The computational complexity of convolution is:
```math
O(H \times W \times F \times F \times C_{in} \times C_{out})
```

### Memory Usage

Memory usage for convolution:
```math
\text{Memory} = O(H \times W \times C_{in}) + O(F \times F \times C_{in} \times C_{out})
```

### Optimization Techniques

1. **FFT-based convolution**: For large kernels
2. **Winograd convolution**: For small kernels (3x3, 5x5)
3. **Im2col + GEMM**: Convert convolution to matrix multiplication
4. **CuDNN**: GPU-optimized convolution algorithms

## Summary

Convolutional operations are the core of CNNs, providing:
- **Efficient feature extraction** through local connectivity
- **Parameter sharing** for translation invariance
- **Hierarchical feature learning** through multiple layers
- **Flexible architecture** with various kernel sizes, strides, and padding options

Understanding these operations is fundamental to designing and implementing effective CNN architectures. 