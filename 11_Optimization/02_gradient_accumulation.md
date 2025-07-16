# Gradient Accumulation

Gradient accumulation is a technique that allows training with large effective batch sizes even when memory constraints limit the actual batch size that can fit in GPU memory.

> **Key Insight:** Gradient accumulation lets you train with large batch sizes on limited hardware, improving convergence and stability without needing massive GPUs.

> **Did you know?** Many state-of-the-art models (e.g., BERT, GPT) are trained with very large effective batch sizes using gradient accumulation!

## Overview

In deep learning, larger batch sizes often lead to more stable training and better convergence. However, GPU memory limitations prevent using very large batch sizes directly. Gradient accumulation solves this by accumulating gradients over multiple forward/backward passes before performing a parameter update, effectively simulating a larger batch size.

> **Geometric Intuition:** Imagine a group of people each carrying a small bucket of water to fill a large tank. Individually, they can't carry much, but together, after several trips, the tank is full. Gradient accumulation works the same way for gradients.

## Mathematical Foundation

### Standard Gradient Descent
In standard gradient descent with batch size $`B`$:
```math
\theta_{t+1} = \theta_t - \alpha \cdot \frac{1}{B} \sum_{i=1}^{B} \nabla L(x_i, y_i; \theta_t)
```

### Gradient Accumulation
With gradient accumulation over $`N`$ accumulation steps:
```math
\theta_{t+1} = \theta_t - \alpha \cdot \frac{1}{N \cdot B} \sum_{k=1}^{N} \sum_{i=1}^{B} \nabla L(x_i^{(k)}, y_i^{(k)}; \theta_t)
```

### Effective Batch Size
The effective batch size is:
```math
B_{\text{effective}} = B_{\text{local}} \times N_{\text{accumulation}}
```

> **Common Pitfall:** Forgetting to scale the loss by the number of accumulation steps can lead to incorrect gradient magnitudes and unstable training.

## Implementation Strategies

### 1. Basic Gradient Accumulation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class GradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, data, target, step):
        # Forward pass
        output = self.model(data)
        loss = self.criterion(output, target)
        
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
```
*This trainer accumulates gradients over several mini-batches before updating the model parameters.*

### 2. Advanced Gradient Accumulation with Mixed Precision

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

class AMPGradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.scaler = GradScaler()
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, data, target, step):
        # Forward pass with mixed precision
        with autocast():
            output = self.model(data)
            loss = self.criterion(output, target)
            loss = loss / self.accumulation_steps
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        return loss.item()
```
*This trainer combines gradient accumulation with mixed precision for efficient and stable training on modern GPUs.*

### 3. Gradient Accumulation with Learning Rate Scaling

```python
class ScaledGradientAccumulationTrainer:
    def __init__(self, model, optimizer, accumulation_steps=4, base_lr=0.001):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.base_lr = base_lr
        self.criterion = nn.CrossEntropyLoss()
        
        # Scale learning rate for effective batch size
        self.scaled_lr = base_lr * accumulation_steps
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scaled_lr
        
    def train_step(self, data, target, step):
        # Forward pass
        output = self.model(data)
        loss = self.criterion(output, target)
        
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item()
    
    def get_effective_batch_size(self, local_batch_size):
        return local_batch_size * self.accumulation_steps
```
*Scaling the learning rate with the effective batch size can help maintain stable training dynamics.*

## Learning Rate Scaling

### Linear Scaling Rule
When using gradient accumulation, the learning rate should be scaled according to the effective batch size:

```math
\text{LR}_{\text{scaled}} = \text{LR}_{\text{base}} \times \frac{B_{\text{effective}}}{B_{\text{reference}}}
```

> **Try it yourself!** Experiment with different accumulation steps and learning rate scaling. How does it affect convergence and final accuracy?

## Memory Management

### Memory Usage Analysis

```python
import torch
import psutil
import GPUtil

def analyze_memory_usage(model, batch_size, accumulation_steps):
    """Analyze memory usage with different accumulation strategies."""
    
    # Memory without accumulation
    torch.cuda.empty_cache()
    model.train()
    
    # Simulate forward pass
    dummy_input = torch.randn(batch_size * accumulation_steps, 784).cuda()
    dummy_target = torch.randint(0, 10, (batch_size * accumulation_steps,)).cuda()
    
    # Memory before forward pass
    memory_before = torch.cuda.memory_allocated() / 1024**3
    
    # Forward pass
    output = model(dummy_input)
    loss = torch.nn.CrossEntropyLoss()(output, dummy_target)
    
    # Memory after forward pass
    memory_after_forward = torch.cuda.memory_allocated() / 1024**3
    
    # Backward pass
    loss.backward()
    
    # Memory after backward pass
    memory_after_backward = torch.cuda.memory_allocated() / 1024**3
    
    print(f"Memory before forward: {memory_before:.2f} GB")
    print(f"Memory after forward: {memory_after_forward:.2f} GB")
    print(f"Memory after backward: {memory_after_backward:.2f} GB")
```
*This function helps you analyze how gradient accumulation affects memory usage during training.*

---

> **Key Insight:** Gradient accumulation is a practical tool for scaling up training on limited hardware, and is widely used in both research and industry. 