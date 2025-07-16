# Mixed Precision Training

Mixed precision training is a technique that uses both 16-bit and 32-bit floating-point types to reduce memory usage and accelerate training while maintaining model accuracy.

## Overview

Traditional deep learning training uses 32-bit floating-point (FP32) for all computations. However, modern hardware (especially NVIDIA GPUs with Tensor Cores) can perform 16-bit operations much faster while using significantly less memory. Mixed precision training leverages this by using FP16 for most operations while keeping FP32 for critical computations that require higher precision.

## Why Mixed Precision?

### Memory Benefits
- **FP32**: 4 bytes per parameter
- **FP16**: 2 bytes per parameter
- **Memory Savings**: Approximately 50% reduction in memory usage

### Speed Benefits
- **Tensor Cores**: NVIDIA GPUs (Volta, Turing, Ampere) have specialized hardware for FP16 operations
- **Bandwidth**: Reduced memory bandwidth requirements
- **Throughput**: 2-3x faster training on compatible hardware

## Mathematical Foundation

### Floating Point Representation

#### FP32 Format
```math
\text{FP32} = (-1)^s \times 2^{e-127} \times (1 + m)
```
where:
- $`s`$ = sign bit (1 bit)
- $`e`$ = exponent (8 bits, biased by 127)
- $`m`$ = mantissa (23 bits)

#### FP16 Format
```math
\text{FP16} = (-1)^s \times 2^{e-15} \times (1 + m)
```
where:
- $`s`$ = sign bit (1 bit)
- $`e`$ = exponent (5 bits, biased by 15)
- $`m`$ = mantissa (10 bits)

### Dynamic Range Comparison

| Format | Min Positive | Max Value | Precision |
|--------|-------------|-----------|-----------|
| FP32   | 1.18e-38    | 3.4e+38   | ~7 digits |
| FP16   | 6.0e-8      | 65504     | ~3 digits |

## Numerical Challenges

### Underflow
When gradients become too small for FP16 representation:
```math
\text{Underflow occurs when } |g| < 6.0 \times 10^{-8}
```

### Overflow
When gradients become too large for FP16 representation:
```math
\text{Overflow occurs when } |g| > 65504
```

### Loss Scaling
To prevent underflow, we scale the loss by a factor $`S`$:
```math
L_{\text{scaled}} = L \times S
```

The gradients are then scaled back:
```math
\nabla L = \frac{\nabla L_{\text{scaled}}}{S}
```

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
            if self.consecutive_overflows == 0:
                self.scale = min(self.scale * self.scale_factor, 2**15)
```

### 2. Master Weights

```python
class MasterWeightsOptimizer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.master_weights = {}
        
        # Create FP32 master weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.master_weights[name] = param.data.clone().float()
    
    def step(self):
        # Update master weights in FP32
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                master_param = self.master_weights[name]
                master_param.add_(param.grad.data.float(), alpha=-self.optimizer.param_groups[0]['lr'])
                param.data = master_param.half()
```

## Best Practices

### 1. Model Architecture Considerations

```python
# Good: Use batch normalization (helps with numerical stability)
class StableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Avoid: Models with very small or large weights
class UnstableModel(nn.Module):
    def __init__(self):
        super().__init__()
        # This can cause numerical issues in FP16
        self.fc = nn.Linear(784, 10)
        nn.init.uniform_(self.fc.weight, -1e-6, 1e-6)
```

### 2. Loss Function Considerations

```python
# Good: Use stable loss functions
criterion = nn.CrossEntropyLoss()

# Avoid: Custom loss functions that might overflow
def unstable_loss(pred, target):
    # This can cause overflow in FP16
    return torch.exp(pred - target).mean()
```

### 3. Gradient Clipping

```python
def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent overflow in mixed precision training."""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
```

## Performance Monitoring

### 1. Memory Usage Tracking

```python
import torch
import psutil
import GPUtil

def monitor_memory():
    """Monitor CPU and GPU memory usage."""
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    print(f"CPU Memory: {cpu_memory.percent}% used")
    
    # GPU memory
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% used")
        
    # PyTorch memory
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"PyTorch GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

# Usage in training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx % 100 == 0:
            monitor_memory()
        # ... training code
```

### 2. Numerical Stability Monitoring

```python
def check_numerical_stability(model):
    """Check for NaN or Inf values in model parameters and gradients."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}")
            if torch.isinf(param.grad).any():
                print(f"Inf detected in gradients of {name}")
        
        if torch.isnan(param.data).any():
            print(f"NaN detected in parameters of {name}")
        if torch.isinf(param.data).any():
            print(f"Inf detected in parameters of {name}")
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# Model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Initialize model, optimizer, and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()
criterion = nn.NLLLoss()

# Training loop with mixed precision
def train_epoch():
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# Training
num_epochs = 5
for epoch in range(num_epochs):
    train_epoch()
    print(f'Epoch {epoch + 1} completed')

print('Training finished!')
```

## Troubleshooting

### Common Issues and Solutions

1. **Loss becomes NaN**
   - Reduce learning rate
   - Use gradient clipping
   - Check for numerical instability in loss function

2. **Model doesn't converge**
   - Ensure loss scaling is working correctly
   - Use master weights for critical layers
   - Monitor gradient norms

3. **Memory usage not reduced**
   - Verify AMP is properly enabled
   - Check that model parameters are in FP16
   - Monitor memory allocation patterns

4. **Performance not improved**
   - Ensure hardware supports Tensor Cores
   - Check that operations are actually using FP16
   - Profile with tools like `torch.profiler`

## Summary

Mixed precision training is a powerful technique that can significantly reduce memory usage and accelerate training on modern hardware. The key is to:

1. Use automatic mixed precision frameworks when possible
2. Implement proper loss scaling to prevent underflow
3. Monitor numerical stability throughout training
4. Use gradient clipping and master weights for stability
5. Profile performance to ensure benefits are realized

With proper implementation, mixed precision training can provide 2-3x speedup and 50% memory reduction while maintaining model accuracy. 