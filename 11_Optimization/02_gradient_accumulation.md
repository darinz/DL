# Gradient Accumulation

Gradient accumulation is a technique that allows training with large effective batch sizes even when memory constraints limit the actual batch size that can fit in GPU memory.

## Overview

In deep learning, larger batch sizes often lead to more stable training and better convergence. However, GPU memory limitations prevent using very large batch sizes directly. Gradient accumulation solves this by accumulating gradients over multiple forward/backward passes before performing a parameter update, effectively simulating a larger batch size.

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

# Usage
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = GradientAccumulationTrainer(model, optimizer, accumulation_steps=4)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = trainer.train_step(data, target, batch_idx)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.4f}')
```

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

# Usage with mixed precision
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
trainer = AMPGradientAccumulationTrainer(model, optimizer, accumulation_steps=4)
```

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

## Learning Rate Scaling

### Linear Scaling Rule
When using gradient accumulation, the learning rate should be scaled according to the effective batch size:

```math
\text{LR}_{\text{scaled}} = \text{LR}_{\text{base}} \times \frac{B_{\text{effective}}}{B_{\text{reference}}}
```

### Implementation

```python
def calculate_scaled_learning_rate(base_lr, reference_batch_size, effective_batch_size):
    """Calculate learning rate scaled for effective batch size."""
    return base_lr * (effective_batch_size / reference_batch_size)

def warmup_learning_rate(optimizer, step, warmup_steps, base_lr, scaled_lr):
    """Implement learning rate warmup."""
    if step < warmup_steps:
        lr = base_lr + (scaled_lr - base_lr) * (step / warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# Example usage
base_lr = 0.001
reference_batch_size = 256
local_batch_size = 64
accumulation_steps = 4
effective_batch_size = local_batch_size * accumulation_steps

scaled_lr = calculate_scaled_learning_rate(
    base_lr, reference_batch_size, effective_batch_size
)

print(f"Base LR: {base_lr}")
print(f"Effective batch size: {effective_batch_size}")
print(f"Scaled LR: {scaled_lr}")
```

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
    print(f"Peak memory usage: {memory_after_backward:.2f} GB")
    
    return memory_after_backward

def compare_memory_strategies(model, batch_size, accumulation_steps):
    """Compare memory usage with and without gradient accumulation."""
    
    print("=== Without Gradient Accumulation ===")
    memory_no_accum = analyze_memory_usage(model, batch_size * accumulation_steps, 1)
    
    print("\n=== With Gradient Accumulation ===")
    memory_with_accum = analyze_memory_usage(model, batch_size, accumulation_steps)
    
    print(f"\nMemory savings: {(memory_no_accum - memory_with_accum) / memory_no_accum * 100:.1f}%")
```

## Advanced Techniques

### 1. Dynamic Accumulation Steps

```python
class DynamicGradientAccumulation:
    def __init__(self, model, optimizer, target_batch_size, max_memory_usage=0.8):
        self.model = model
        self.optimizer = optimizer
        self.target_batch_size = target_batch_size
        self.max_memory_usage = max_memory_usage
        self.criterion = nn.CrossEntropyLoss()
        
    def calculate_optimal_batch_size(self):
        """Calculate optimal local batch size based on available memory."""
        total_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = total_memory * self.max_memory_usage
        
        # Estimate memory per sample (this would need to be calibrated)
        memory_per_sample = 0.001  # GB per sample (example value)
        
        optimal_local_batch_size = int(available_memory / memory_per_sample)
        accumulation_steps = max(1, self.target_batch_size // optimal_local_batch_size)
        
        return optimal_local_batch_size, accumulation_steps
    
    def train_step(self, data, target, step, accumulation_steps):
        # Forward pass
        output = self.model(data)
        loss = self.criterion(output, target)
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        return loss.item()
```

### 2. Gradient Accumulation with Validation

```python
class GradientAccumulationWithValidation:
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, val_loader, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            loss = loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Statistics
            running_loss += loss.item() * self.accumulation_steps
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Validation every N accumulation steps
                if (batch_idx + 1) % (self.accumulation_steps * 10) == 0:
                    val_loss, val_acc = self.validate(val_loader)
                    print(f'Epoch {epoch}, Batch {batch_idx}, '
                          f'Train Loss: {running_loss/total:.4f}, '
                          f'Train Acc: {100.*correct/total:.2f}%, '
                          f'Val Loss: {val_loss:.4f}, '
                          f'Val Acc: {val_acc:.2f}%')
        
        return running_loss / total, 100. * correct / total
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return val_loss / len(val_loader), 100. * correct / total
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
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
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

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

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)

# Gradient accumulation parameters
local_batch_size = 32
accumulation_steps = 4
effective_batch_size = local_batch_size * accumulation_steps

# Learning rate scaling
base_lr = 0.001
reference_batch_size = 256
scaled_lr = base_lr * (effective_batch_size / reference_batch_size)

optimizer = optim.Adam(model.parameters(), lr=scaled_lr)
criterion = nn.NLLLoss()

# Training loop with gradient accumulation
def train_epoch():
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Statistics
        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            # Print statistics
            if (i + 1) % (accumulation_steps * 10) == 0:
                print(f'[{epoch + 1}, {i + 1}] '
                      f'loss: {running_loss / total:.3f}, '
                      f'acc: {100. * correct / total:.2f}%')
                running_loss = 0.0
                correct = 0
                total = 0

# Validation
def validate():
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return val_loss / len(testloader), 100. * correct / total

# Training
num_epochs = 5
for epoch in range(num_epochs):
    train_epoch()
    val_loss, val_acc = validate()
    print(f'Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, '
          f'Validation Accuracy: {val_acc:.2f}%')

print('Training finished!')
```

## Best Practices

### 1. Learning Rate Scheduling

```python
def create_scheduler(optimizer, num_training_steps, warmup_steps=0):
    """Create learning rate scheduler for gradient accumulation."""
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                  float(max(1, num_training_steps - warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)

# Usage
num_training_steps = len(trainloader) * num_epochs // accumulation_steps
scheduler = create_scheduler(optimizer, num_training_steps, warmup_steps=1000)

# In training loop
if (i + 1) % accumulation_steps == 0:
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### 2. Gradient Clipping

```python
def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent explosion."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

# In training loop
if (i + 1) % accumulation_steps == 0:
    clip_gradients(model, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

### 3. Monitoring Effective Batch Size

```python
def monitor_effective_batch_size(local_batch_size, accumulation_steps):
    """Monitor and log effective batch size."""
    effective_batch_size = local_batch_size * accumulation_steps
    print(f"Local batch size: {local_batch_size}")
    print(f"Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    return effective_batch_size
```

## Troubleshooting

### Common Issues and Solutions

1. **Training becomes unstable**
   - Reduce learning rate scaling factor
   - Use gradient clipping
   - Increase accumulation steps gradually

2. **Memory usage still too high**
   - Reduce local batch size
   - Increase accumulation steps
   - Use gradient checkpointing

3. **Slow training**
   - Profile memory usage to find optimal batch size
   - Use mixed precision training
   - Optimize data loading

4. **Convergence issues**
   - Ensure proper learning rate scaling
   - Use warmup scheduling
   - Monitor gradient norms

## Summary

Gradient accumulation is a powerful technique that allows training with large effective batch sizes while respecting memory constraints. Key points:

1. **Effective batch size** = Local batch size × Accumulation steps
2. **Learning rate scaling** is crucial for maintaining convergence
3. **Memory efficiency** allows training larger models
4. **Implementation** requires careful gradient scaling and timing
5. **Monitoring** effective batch size and learning rate is essential

When properly implemented, gradient accumulation can significantly improve training stability and convergence while working within hardware limitations. 