# Modern Architectures

Modern CNN architectures focus on efficiency, scalability, and deployment on resource-constrained devices. These architectures introduce novel design principles to achieve better performance with fewer parameters and computational resources.

> **Key Insight:**
> 
> The latest CNN architectures are not just about accuracy—they're about doing more with less. Efficiency and deployability are now as important as raw power.

## Table of Contents

1. [EfficientNet (2019)](#efficientnet-2019)
2. [MobileNet (2017)](#mobilenet-2017)
3. [ShuffleNet (2017)](#shufflenet-2017)
4. [Design Principles](#design-principles)
5. [Performance Comparison](#performance-comparison)
6. [Summary Table](#summary-table)
7. [Actionable Next Steps](#actionable-next-steps)

---

## EfficientNet (2019)

### Historical Context

EfficientNet, developed by Google Research, introduced compound scaling that uniformly scales network depth, width, and resolution using a compound coefficient. It achieved state-of-the-art accuracy with significantly fewer parameters.

> **Explanation:**
> EfficientNet's main innovation is to scale all dimensions of a network (depth, width, and input resolution) together in a principled way, rather than arbitrarily. This leads to better accuracy and efficiency.

> **Did you know?**
> EfficientNet models are widely used in Kaggle competitions and real-world applications due to their excellent accuracy/efficiency trade-off.

### Compound Scaling Method

The key innovation is compound scaling, which scales all three dimensions (depth, width, resolution) together:

```math
\text{depth}: d = \alpha^\phi
\text{width}: w = \beta^\phi  
\text{resolution}: r = \gamma^\phi
\text{where } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
```

Where $`\phi`$ is the compound coefficient that controls resource scaling.

> **Math Breakdown:**
> - $\alpha, \beta, \gamma$ are constants that determine how much to scale each dimension.
> - $\phi$ is a user-chosen parameter that increases model size and accuracy.
> - The constraint $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ ensures balanced scaling.

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

> **Explanation:**
> These equations show how the number of layers, channels, and input size are scaled at each stage. The total computational cost (FLOPs) grows as all three are increased.

> **Key Insight:**
> Compound scaling lets you "dial up" or "dial down" the model for your hardware, keeping the architecture balanced.

---

## MobileNet (2017)

### Historical Context

MobileNet, developed by Google, introduced depthwise separable convolutions to create efficient neural networks for mobile and embedded vision applications.

> **Explanation:**
> MobileNet's main innovation is to break standard convolution into two simpler operations (depthwise and pointwise), drastically reducing computation and parameters while maintaining accuracy.

> **Try it yourself!**
> Compare the number of parameters and FLOPs in a standard convolution vs. a depthwise separable convolution. How much do you save?

### Depthwise Separable Convolution

The key innovation is decomposing standard convolution into two steps:

**1. Depthwise Convolution:**
```math
(I * K_d)(i, j, c) = \sum_{m,n} I(i+m, j+n, c) \cdot K_d(m, n, c)
```

> **Math Breakdown:**
> - Each input channel is convolved with its own filter (no mixing between channels).

**2. Pointwise Convolution:**
```math
(F * K_p)(i, j, k) = \sum_{c} F(i, j, c) \cdot K_p(c, k)
```

> **Math Breakdown:**
> - A $1 \times 1$ convolution mixes the output of the depthwise step across channels, allowing for feature combination.

**Computational Reduction:**
```math
\text{Standard Conv}: O(H \times W \times F \times F \times C_{in} \times C_{out})
\text{Depthwise Separable}: O(H \times W \times F \times F \times C_{in} + H \times W \times C_{in} \times C_{out})
```

> **Explanation:**
> Depthwise separable convolution reduces computation by splitting spatial and channel mixing into two steps, making it much more efficient for mobile devices.

> **Key Insight:**
> Depthwise separable convolution is the secret sauce behind many mobile-optimized models.

---

## ShuffleNet (2017)

### Historical Context

ShuffleNet, developed by Megvii Inc., introduced channel shuffling to enable efficient group convolutions while maintaining accuracy.

> **Explanation:**
> ShuffleNet's innovation is to use group convolutions (which are efficient but can block information flow) and then shuffle the channels so that information can mix between groups.

> **Did you know?**
> ShuffleNet's channel shuffle operation ensures that information can flow between groups, overcoming a key limitation of group convolutions.

### Channel Shuffling

The key innovation is channel shuffling, which enables information flow between groups:

```math
\text{Shuffle}(x) = \text{Reshape}(\text{Transpose}(\text{Reshape}(x, g, c/g, h, w), 1, 2), c, h, w)
```

Where $`g`$ is the number of groups and $`c`$ is the number of channels.

> **Math Breakdown:**
> - The input tensor is reshaped to separate groups, transposed to mix channels, and then reshaped back.
> - This operation allows features from different groups to interact, improving accuracy.

> **Try it yourself!**
> Implement a group convolution with and without channel shuffling. Compare the accuracy on a small dataset.

---

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

> **Key Insight:**
> 
> Modern architectures are built from modular blocks—mix and match these patterns to design your own efficient networks!

---

## Performance Comparison

### Accuracy vs Efficiency Trade-off

| Architecture      | Top-1 Accuracy | Parameters (M) | FLOPs (M) |
|------------------|----------------|----------------|-----------|
| EfficientNet-B0  | 77.1%          | 5.3            | 390       |
| MobileNet-v1     | 70.6%          | 4.2            | 569       |
| ShuffleNet-v1    | 67.4%          | 1.9            | 140       |

### Deployment Considerations

**Mobile Devices:**
- **Latency**: Real-time inference requirements
- **Memory**: Limited RAM constraints
- **Battery**: Power consumption optimization

**Edge Devices:**
- **Model size**: Storage limitations
- **Computational resources**: Limited processing power
- **Network connectivity**: Offline operation capability

---

## Summary Table

| Architecture      | Year | Key Innovation(s)                | Top-1 Acc. | Params (M) | FLOPs (M) | Best Use Case         |
|------------------|------|----------------------------------|------------|------------|-----------|----------------------|
| EfficientNet-B0  | 2019 | Compound scaling, MBConv, SE     | 77.1%      | 5.3        | 390       | Mobile, cloud        |
| MobileNet-v1     | 2017 | Depthwise separable conv         | 70.6%      | 4.2        | 569       | Mobile, embedded     |
| ShuffleNet-v1    | 2017 | Channel shuffle, group conv      | 67.4%      | 1.9        | 140       | Mobile, edge         |

---

## Actionable Next Steps

- **Experiment:** Try training a MobileNet, ShuffleNet, and EfficientNet on a small dataset. Compare speed, accuracy, and model size.
- **Visualize:** Plot the feature maps and activation distributions for each architecture.
- **Diagnose:** If your model is too slow or large for your device, try using a width multiplier or compound scaling.
- **Connect:** See how these architectures inspire the design of vision transformers and hybrid models in later chapters.

> **Key Insight:**
> 
> The future of deep learning is efficient, scalable, and everywhere—from your phone to the cloud. Master these architectures to build the next generation of AI applications! 