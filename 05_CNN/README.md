# Convolutional Neural Networks (CNNs)

Convolutional Neural Networks are specialized neural networks designed for processing grid-like data, particularly images. They leverage the principles of local connectivity, shared weights, and spatial hierarchies to efficiently learn hierarchical feature representations.

## Table of Contents

1. [Convolutional Operations](#convolutional-operations) - [Detailed Guide](01_convolutional_operations.md)
2. [Pooling Layers](#pooling-layers) - [Detailed Guide](02_pooling_layers.md)
3. [Architecture Evolution](#architecture-evolution) - [Detailed Guide](03_architecture_evolution.md)
4. [Modern Architectures](#modern-architectures) - [Detailed Guide](04_modern_architectures.md)
5. [Computer Vision Tasks](#computer-vision-tasks) - [Detailed Guide](05_computer_vision_tasks.md)

## Convolutional Operations

Convolutional operations are the core building blocks of CNNs, performing feature extraction through sliding filters.

### Basic Convolution

The discrete convolution operation is defined as:

```math
(I * K)(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) \cdot K(m, n)
```

Where:
- $`I`$ is the input feature map
- $`K`$ is the convolutional kernel/filter
- $`(i, j)`$ are the output coordinates

### Key Properties

**Local Connectivity**: Each neuron only connects to a local region of the input, reducing parameters and computational complexity.

**Weight Sharing**: The same filter is applied across the entire input, enabling translation invariance and parameter efficiency.

**Feature Maps**: Multiple filters create different feature maps, each detecting specific patterns.

### Stride and Padding

**Stride** $`s`$: Controls the step size of the filter:
```math
\text{Output Size} = \left\lfloor \frac{n - f + 2p}{s} \right\rfloor + 1
```

**Padding** $`p`$: Adds zeros around the input to control output size:
- **Valid padding**: No padding, output size decreases
- **Same padding**: Output size equals input size

## Pooling Layers

Pooling layers reduce spatial dimensions while preserving important features.

### Max Pooling

Selects the maximum value in each pooling window:

```math
y_{i,j} = \max_{(m,n) \in R_{i,j}} x_{m,n}
```

### Average Pooling

Computes the average value in each pooling window:

```math
y_{i,j} = \frac{1}{|R_{i,j}|} \sum_{(m,n) \in R_{i,j}} x_{m,n}
```

### Global Pooling

Reduces spatial dimensions to a single value:
- **Global Average Pooling**: $`y = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{i,j}`$
- **Global Max Pooling**: $`y = \max_{i,j} x_{i,j}`$

## Architecture Evolution

### LeNet-5 (1998)
- First successful CNN for digit recognition
- 7 layers: 2 convolutional + 2 pooling + 3 fully connected
- ReLU activation (before it was popular)

### AlexNet (2012)
- Breakthrough in ImageNet competition
- 8 layers: 5 convolutional + 3 fully connected
- ReLU activation, dropout, data augmentation
- GPU implementation

### VGG (2014)
- Simple architecture with 3x3 convolutions
- Deep networks (16-19 layers)
- Demonstrated depth importance
- Consistent design pattern

### ResNet (2015)
- Introduced residual connections
- Solved vanishing gradient problem
- Residual block: $`F(x) + x`$
- Enabled training of very deep networks (100+ layers)

### DenseNet (2017)
- Dense connections between all layers
- Feature reuse and gradient flow
- Concatenation: $`x_l = H_l([x_0, x_1, ..., x_{l-1}])`$

## Modern Architectures

### EfficientNet (2019)
Compound scaling method optimizing depth, width, and resolution:

```math
\text{depth}: d = \alpha^\phi
\text{width}: w = \beta^\phi  
\text{resolution}: r = \gamma^\phi
\text{where } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
```

### MobileNet (2017)
Depthwise separable convolutions for efficiency:
- **Depthwise convolution**: $`(I * K_d)(i, j, c) = \sum_{m,n} I(i+m, j+n, c) \cdot K_d(m, n, c)`$
- **Pointwise convolution**: $`(F * K_p)(i, j, k) = \sum_{c} F(i, j, c) \cdot K_p(c, k)`$

### ShuffleNet (2017)
Channel shuffling for efficient group convolutions:
- Reduces computational cost
- Maintains accuracy
- Mobile-friendly architecture

## Computer Vision Tasks

### Image Classification
- **Input**: Single image
- **Output**: Class probabilities $`P(y|x)`$
- **Loss**: Cross-entropy loss
- **Metrics**: Top-1/Top-5 accuracy

### Object Detection
- **Input**: Image
- **Output**: Bounding boxes + class labels
- **Architectures**: R-CNN, Fast R-CNN, Faster R-CNN, YOLO, SSD
- **Loss**: Classification + regression losses

### Semantic Segmentation
- **Input**: Image
- **Output**: Pixel-wise class labels
- **Architectures**: FCN, U-Net, DeepLab
- **Loss**: Pixel-wise cross-entropy

### Instance Segmentation
- **Input**: Image
- **Output**: Instance masks + class labels
- **Architectures**: Mask R-CNN, YOLACT
- **Combines**: Detection + segmentation

## Key Concepts

### Receptive Field
The region of input that affects a particular output neuron:
```math
RF_l = RF_{l-1} + (k_l - 1) \prod_{i=1}^{l-1} s_i
```

### Feature Hierarchy
- Early layers: Edges, textures, simple patterns
- Middle layers: Parts, shapes, complex patterns
- Late layers: Objects, semantic concepts

### Transfer Learning
- Pre-trained models on large datasets
- Fine-tuning for specific tasks
- Feature extraction for new domains

## Implementation Considerations

### Memory Management
- Gradient checkpointing for large models
- Mixed precision training
- Model parallelism

### Training Strategies
- Learning rate scheduling
- Data augmentation
- Regularization techniques
- Batch normalization

### Deployment
- Model quantization
- Pruning for efficiency
- Mobile optimization
- Edge deployment

## Detailed Guides

For comprehensive explanations, mathematical formulations, and implementation examples, refer to the following detailed guides:

- **[01_convolutional_operations.md](01_convolutional_operations.md)** - Complete guide to convolutional operations with implementations
- **[02_pooling_layers.md](02_pooling_layers.md)** - Detailed coverage of pooling layers and techniques
- **[03_architecture_evolution.md](03_architecture_evolution.md)** - Historical evolution from LeNet to DenseNet
- **[04_modern_architectures.md](04_modern_architectures.md)** - Modern efficient architectures (EfficientNet, MobileNet, ShuffleNet)
- **[05_computer_vision_tasks.md](05_computer_vision_tasks.md)** - Task-specific implementations (classification, detection, segmentation) 