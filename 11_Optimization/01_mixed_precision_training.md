# Mixed Precision Training

Mixed precision training is a technique that uses both 16-bit and 32-bit floating-point types to reduce memory usage and accelerate training while maintaining model accuracy.

> **Explanation:**
> Mixed precision training means using both FP16 (16-bit) and FP32 (32-bit) numbers in your model. This saves memory and can make training much faster, especially on modern GPUs, while still keeping the model accurate.

> **Key Insight:** Mixed precision training enables you to train larger models or use bigger batch sizes without running out of GPU memory.

> **Did you know?** Modern GPUs (like NVIDIA Volta, Turing, Ampere) have specialized Tensor Cores that are optimized for FP16 operations, making mixed precision especially effective.

## Overview

Traditional deep learning training uses 32-bit floating-point (FP32) for all computations. However, modern hardware (especially NVIDIA GPUs with Tensor Cores) can perform 16-bit operations much faster while using significantly less memory. Mixed precision training leverages this by using FP16 for most operations while keeping FP32 for critical computations that require higher precision.

> **Explanation:**
> Most calculations are done in FP16 for speed and memory savings, but some sensitive operations (like loss calculation and weight updates) are kept in FP32 to avoid numerical issues.

## Why Mixed Precision?

### Memory Benefits
- **FP32**: 4 bytes per parameter
- **FP16**: 2 bytes per parameter
- **Memory Savings**: Approximately 50% reduction in memory usage

> **Explanation:**
> Using FP16 halves the memory needed for model parameters, activations, and gradients, allowing you to train larger models or use bigger batches.

### Speed Benefits
- **Tensor Cores**: NVIDIA GPUs (Volta, Turing, Ampere) have specialized hardware for FP16 operations
- **Bandwidth**: Reduced memory bandwidth requirements
- **Throughput**: 2-3x faster training on compatible hardware

> **Explanation:**
> FP16 operations are much faster on supported GPUs, and less data needs to be moved around, speeding up training.

> **Geometric Intuition:** Imagine carrying water in buckets (FP32) vs. cups (FP16). You can carry more cups at once, but each holds less water. Mixed precision lets you use both, optimizing for speed and capacity.

## Mathematical Foundation

### Floating Point Representation

#### FP32 Format
```math
\text{FP32} = (-1)^s \times 2^{e-127} \times (1 + m)
```
> **Math Breakdown:**
> - $`s`$: Sign bit (1 bit)
> - $`e`$: Exponent (8 bits, biased by 127)
> - $`m`$: Mantissa (23 bits)
> - FP32 can represent a wide range of values with high precision.

#### FP16 Format
```math
\text{FP16} = (-1)^s \times 2^{e-15} \times (1 + m)
```
> **Math Breakdown:**
> - $`s`$: Sign bit (1 bit)
> - $`e`$: Exponent (5 bits, biased by 15)
> - $`m`$: Mantissa (10 bits)
> - FP16 uses less memory but has a smaller range and lower precision than FP32.

### Dynamic Range Comparison

| Format | Min Positive | Max Value | Precision |
|--------|-------------|-----------|-----------|
| FP32   | 1.18e-38    | 3.4e+38   | ~7 digits |
| FP16   | 6.0e-8      | 65504     | ~3 digits |

> **Explanation:**
> FP16 can't represent very small or very large numbers as well as FP32, and is less precise. This can cause problems if not handled carefully.

> **Common Pitfall:** FP16 has a much smaller dynamic range and lower precision than FP32, making it more susceptible to underflow and overflow.

## Numerical Challenges

### Underflow
When gradients become too small for FP16 representation:
```math
\text{Underflow occurs when } |g| < 6.0 \times 10^{-8}
```
> **Math Breakdown:**
> If a gradient is smaller than $6.0 \times 10^{-8}$, it becomes zero in FP16, so the model can't learn from it.

### Overflow
When gradients become too large for FP16 representation:
```math
\text{Overflow occurs when } |g| > 65504
```
> **Math Breakdown:**
> If a gradient is larger than 65504, it becomes infinity in FP16, which can break training.

### Loss Scaling
To prevent underflow, we scale the loss by a factor $`S`$:
```math
L_{\text{scaled}} = L \times S
```
> **Explanation:**
> By multiplying the loss by a large number, we make the gradients bigger so they don't vanish in FP16.

The gradients are then scaled back:
```math
\nabla L = \frac{\nabla L_{\text{scaled}}}{S}
```
> **Explanation:**
> After computing gradients, we divide by the same scale factor to get the correct values.

> **Key Insight:** Loss scaling is essential for stable mixed precision training, as it prevents small gradients from vanishing in FP16.

## Implementation Strategies

### 1. Manual Mixed Precision

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ManualMixedPrecision:
    def __init__(self, model, optimizer, loss_scale=2**15):
        self.model = model
        self.optimizer = optimizer
        self.loss_scale = loss_scale
        self.scaler = torch.cuda.amp.GradScaler()
        
    def forward(self, x):
        # Convert input to FP16
        x = x.half()
        
        # Forward pass in FP16
        with torch.cuda.amp.autocast():
            output = self.model(x)
            
        return output
    
    def backward(self, loss):
        # Scale loss to prevent underflow
        scaled_loss = loss * self.loss_scale
        
        # Backward pass
        scaled_loss.backward()
        
        # Unscale gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.data /= self.loss_scale
                
        # Check for overflow
        overflow = False
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                    overflow = True
                    break
        
        return overflow
    
    def step(self, loss):
        overflow = self.backward(loss)
        
        if not overflow:
            self.optimizer.step()
        else:
            # Skip update if overflow occurred
            print("Gradient overflow detected, skipping update")
            
        self.optimizer.zero_grad()
```
> **Code Walkthrough:**
> - Converts input to FP16 for the forward pass.
> - Scales the loss to avoid underflow, then unscales gradients after backward pass.
> - Checks for overflow (infinite or NaN gradients) and skips the optimizer step if detected.
> - This manual approach is educational, but most users should use PyTorch AMP for simplicity and safety.

*This class demonstrates manual loss scaling and overflow detection for mixed precision training.*

### 2. PyTorch Automatic Mixed Precision (AMP)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class AMPTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def train_step(self, data, target):
        self.optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        with autocast():
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Optimizer step with unscaling
        self.scaler.step(self.optimizer)
        
        # Update scaler for next iteration
        self.scaler.update()
        
        return loss.item()

# Usage example
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = AMPTrainer(model, optimizer)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        loss = trainer.train_step(data, target)
```
> **Code Walkthrough:**
> - Uses PyTorch's AMP to automatically handle mixed precision and loss scaling.
> - The `autocast` context manager runs operations in FP16 where safe, and FP32 where needed.
> - The `GradScaler` handles scaling and unscaling of gradients, and skips updates if overflow is detected.
> - This is the recommended way to use mixed precision in PyTorch.

*PyTorch AMP automates most of the mixed precision workflow, making it easy and robust.*

### 3. TensorFlow Mixed Precision

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Create model
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# Compile with mixed precision
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training automatically uses mixed precision
model.fit(x_train, y_train, epochs=10, batch_size=32)
```
> **Code Walkthrough:**
> - Sets the global policy to use mixed precision in TensorFlow.
> - All layers and computations use FP16 where possible.
> - Training and evaluation are handled as usual, but with the benefits of mixed precision.

*TensorFlow's mixed precision API is simple to use and can provide significant speedups on supported hardware.*

## Advanced Techniques

### 1. Dynamic Loss Scaling

```python
class DynamicLossScaler:
    def __init__(self, init_scale=2**15, scale_factor=2, scale_window=2000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.consecutive_overflows = 0
        
    def scale_loss(self, loss):
        return loss * self.scale
    
    def unscale_gradients(self, gradients):
        return [g / self.scale for g in gradients]
    
    def update_scale(self, overflow_detected):
        if overflow_detected:
            self.consecutive_overflows += 1
            if self.consecutive_overflows >= self.scale_window:
                self.scale = max(self.scale / self.scale_factor, 1)
                self.consecutive_overflows = 0
        else:
            self.consecutive_overflows = 0
```
> **Code Walkthrough:**
> - Dynamically adjusts the loss scale based on whether overflow is detected.
> - If overflows happen too often, the scale is reduced; otherwise, it is kept or increased.
> - This helps maintain stability and efficiency during training. 