# Knowledge Distillation

Knowledge distillation is a technique for transferring knowledge from a large, complex model (teacher) to a smaller, simpler model (student) while maintaining or improving performance.

## Overview

Knowledge distillation enables:
- **Model compression** for deployment on resource-constrained devices
- **Performance improvement** through teacher guidance
- **Knowledge transfer** from ensemble models
- **Privacy preservation** by training on soft targets

## Mathematical Foundation

### Soft Targets and Temperature Scaling

The teacher's soft predictions are computed using temperature scaling:
```math
q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
```

where:
- $`z_i`$ = logit for class $`i`$
- $`T`$ = temperature parameter
- $`q_i`$ = soft probability for class $`i`$

### Knowledge Distillation Loss

The total loss combines hard and soft targets:
```math
L = \alpha L_{\text{hard}} + (1 - \alpha) L_{\text{soft}}
```

where:
- $`L_{\text{hard}}`$ = standard cross-entropy loss with hard labels
- $`L_{\text{soft}}`$ = KL divergence loss with soft targets
- $`\alpha`$ = weight balancing parameter

### KL Divergence Loss

```math
L_{\text{soft}} = T^2 \cdot \text{KL}(q^T \| p^T)
```

where:
- $`q^T`$ = teacher's soft predictions
- $`p^T`$ = student's soft predictions
- $`T^2`$ = temperature scaling factor

## Implementation Strategies

### 1. Basic Knowledge Distillation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class KnowledgeDistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
        
    def distillation_loss(self, student_logits, teacher_logits, targets):
        """Compute knowledge distillation loss."""
        # Hard loss (standard cross-entropy)
        hard_loss = F.cross_entropy(student_logits, targets)
        
        # Soft loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss
    
    def train_step(self, data, targets, optimizer):
        """Single training step."""
        optimizer.zero_grad()
        
        # Forward pass through teacher and student
        with torch.no_grad():
            teacher_logits = self.teacher_model(data)
        
        student_logits = self.student_model(data)
        
        # Compute distillation loss
        total_loss, hard_loss, soft_loss = self.distillation_loss(
            student_logits, teacher_logits, targets
        )
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), hard_loss.item(), soft_loss.item()

# Usage example
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize models
teacher_model = TeacherModel()
student_model = StudentModel()

# Load pre-trained teacher (or train it first)
# teacher_model.load_state_dict(torch.load('teacher_model.pth'))

# Initialize trainer
trainer = KnowledgeDistillationTrainer(teacher_model, student_model)

# Training loop
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        total_loss, hard_loss, soft_loss = trainer.train_step(data, targets, optimizer)
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}')
            print(f'Total Loss: {total_loss:.4f}, Hard Loss: {hard_loss:.4f}, Soft Loss: {soft_loss:.4f}')
```

### 2. Advanced Knowledge Distillation with Feature Matching

```python
class FeatureDistillationTrainer:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7, beta=0.3):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def feature_matching_loss(self, student_features, teacher_features):
        """Compute feature matching loss."""
        feature_loss = 0
        for student_feat, teacher_feat in zip(student_features, teacher_features):
            # L2 loss between feature maps
            feature_loss += F.mse_loss(student_feat, teacher_feat)
        return feature_loss
    
    def train_step_with_features(self, data, targets, optimizer):
        """Training step with feature matching."""
        optimizer.zero_grad()
        
        # Forward pass through teacher
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher_model(data, return_features=True)
        
        # Forward pass through student
        student_logits, student_features = self.student_model(data, return_features=True)
        
        # Distillation loss
        total_loss, hard_loss, soft_loss = self.distillation_loss(
            student_logits, teacher_logits, targets
        )
        
        # Feature matching loss
        feature_loss = self.feature_matching_loss(student_features, teacher_features)
        
        # Combined loss
        final_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss + self.beta * feature_loss
        
        final_loss.backward()
        optimizer.step()
        
        return final_loss.item(), hard_loss.item(), soft_loss.item(), feature_loss.item()

# Modified models to return features
class TeacherModelWithFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, return_features=False):
        features = []
        
        x = F.relu(self.fc1(x))
        features.append(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        features.append(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        if return_features:
            return x, features
        return x

class StudentModelWithFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x, return_features=False):
        features = []
        
        x = F.relu(self.fc1(x))
        features.append(x)
        
        x = self.fc2(x)
        
        if return_features:
            return x, features
        return x
```

### 3. Progressive Knowledge Distillation

```python
class ProgressiveDistillationTrainer:
    def __init__(self, teacher_model, student_model, num_stages=3):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.num_stages = num_stages
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def progressive_training(self, train_loader, num_epochs_per_stage=10):
        """Progressive knowledge distillation training."""
        current_teacher = self.teacher_model
        
        for stage in range(self.num_stages):
            print(f"Training stage {stage + 1}/{self.num_stages}")
            
            # Adjust temperature and alpha for each stage
            temperature = 4.0 - stage * 0.5  # Decrease temperature
            alpha = 0.3 + stage * 0.2  # Increase hard loss weight
            
            # Create trainer for this stage
            trainer = KnowledgeDistillationTrainer(
                current_teacher, self.student_model, 
                temperature=temperature, alpha=alpha
            )
            
            # Train for this stage
            optimizer = optim.Adam(self.student_model.parameters(), lr=0.001)
            
            for epoch in range(num_epochs_per_stage):
                for batch_idx, (data, targets) in enumerate(train_loader):
                    total_loss, hard_loss, soft_loss = trainer.train_step(data, targets, optimizer)
                    
                    if batch_idx % 100 == 0:
                        print(f'Stage {stage + 1}, Epoch {epoch}, Batch {batch_idx}')
                        print(f'Loss: {total_loss:.4f}')
            
            # Update teacher for next stage (optional)
            if stage < self.num_stages - 1:
                current_teacher = type(self.student_model)()
                current_teacher.load_state_dict(self.student_model.state_dict())
                current_teacher.eval()
```

## Advanced Techniques

### 1. Attention Transfer

```python
class AttentionTransfer:
    def __init__(self, attention_maps_teacher, attention_maps_student):
        self.attention_maps_teacher = attention_maps_teacher
        self.attention_maps_student = attention_maps_student
    
    def attention_loss(self, student_attention, teacher_attention):
        """Compute attention transfer loss."""
        # Normalize attention maps
        student_attention = F.normalize(student_attention.view(student_attention.size(0), -1), dim=1)
        teacher_attention = F.normalize(teacher_attention.view(teacher_attention.size(0), -1), dim=1)
        
        # Compute attention transfer loss
        attention_loss = F.mse_loss(student_attention, teacher_attention)
        return attention_loss

class AttentionTransferTrainer:
    def __init__(self, teacher_model, student_model, temperature=4.0, alpha=0.7, gamma=0.1):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.teacher_model.eval()
    
    def get_attention_maps(self, model, x):
        """Extract attention maps from model."""
        attention_maps = []
        
        # Hook to capture intermediate activations
        def hook_fn(module, input, output):
            attention_maps.append(output)
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass
        _ = model(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return attention_maps
    
    def train_step_with_attention(self, data, targets, optimizer):
        """Training step with attention transfer."""
        optimizer.zero_grad()
        
        # Get teacher attention maps
        with torch.no_grad():
            teacher_logits = self.teacher_model(data)
            teacher_attention = self.get_attention_maps(self.teacher_model, data)
        
        # Get student attention maps
        student_logits = self.student_model(data)
        student_attention = self.get_attention_maps(self.student_model, data)
        
        # Distillation loss
        total_loss, hard_loss, soft_loss = self.distillation_loss(
            student_logits, teacher_logits, targets
        )
        
        # Attention transfer loss
        attention_loss = 0
        for student_att, teacher_att in zip(student_attention, teacher_attention):
            attention_loss += F.mse_loss(student_att, teacher_att)
        
        # Combined loss
        final_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss + self.gamma * attention_loss
        
        final_loss.backward()
        optimizer.step()
        
        return final_loss.item(), hard_loss.item(), soft_loss.item(), attention_loss.item()
```

### 2. Self-Distillation

```python
class SelfDistillationTrainer:
    def __init__(self, model, temperature=4.0, alpha=0.7):
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.ema_model = type(model)()
        self.ema_model.load_state_dict(model.state_dict())
        
        # Freeze EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False
        
        self.ema_model.eval()
    
    def update_ema_model(self, decay=0.999):
        """Update exponential moving average model."""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data = decay * ema_param.data + (1 - decay) * param.data
    
    def train_step_self_distillation(self, data, targets, optimizer):
        """Self-distillation training step."""
        optimizer.zero_grad()
        
        # Forward pass through main model
        student_logits = self.model(data)
        
        # Forward pass through EMA model
        with torch.no_grad():
            teacher_logits = self.ema_model(data)
        
        # Distillation loss
        total_loss, hard_loss, soft_loss = self.distillation_loss(
            student_logits, teacher_logits, targets
        )
        
        total_loss.backward()
        optimizer.step()
        
        # Update EMA model
        self.update_ema_model()
        
        return total_loss.item(), hard_loss.item(), soft_loss.item()
```

## Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# Teacher model (large)
class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Student model (small)
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training functions
def train_teacher():
    """Train the teacher model."""
    teacher = TeacherModel()
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    
    print("Training teacher model...")
    for epoch in range(10):
        teacher.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    return teacher

def train_student_with_distillation(teacher, student):
    """Train student model with knowledge distillation."""
    trainer = KnowledgeDistillationTrainer(teacher, student, temperature=4.0, alpha=0.7)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    print("Training student model with knowledge distillation...")
    for epoch in range(10):
        student.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            total_loss, hard_loss, soft_loss = trainer.train_step(inputs, labels, optimizer)
            running_loss += total_loss
            
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    return student

def evaluate_model(model, testloader):
    """Evaluate model performance."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Main training process
if __name__ == "__main__":
    # Train teacher model
    teacher_model = train_teacher()
    teacher_accuracy = evaluate_model(teacher_model, testloader)
    print(f"Teacher model accuracy: {teacher_accuracy:.2f}%")
    
    # Train student model with distillation
    student_model = StudentModel()
    student_model = train_student_with_distillation(teacher_model, student_model)
    student_accuracy = evaluate_model(student_model, testloader)
    print(f"Student model accuracy: {student_accuracy:.2f}%")
    
    # Compare model sizes
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    teacher_params = count_parameters(teacher_model)
    student_params = count_parameters(student_model)
    
    print(f"Teacher model parameters: {teacher_params:,}")
    print(f"Student model parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params / student_params:.2f}x")
```

## Best Practices

### 1. Temperature Tuning
- Start with high temperature (4-10) for soft targets
- Gradually decrease temperature during training
- Use validation to find optimal temperature

### 2. Loss Weighting
- Balance hard and soft losses appropriately
- Use curriculum learning for loss weights
- Monitor both losses during training

### 3. Architecture Design
- Design student architecture carefully
- Consider feature matching for better transfer
- Use attention mechanisms when applicable

### 4. Training Strategy
- Use progressive distillation for complex tasks
- Implement self-distillation for further improvement
- Combine with other compression techniques

## Summary

Knowledge distillation is a powerful technique for model compression and performance improvement. Key considerations:

1. **Temperature scaling** controls the softness of teacher guidance
2. **Loss balancing** between hard and soft targets is crucial
3. **Feature matching** can improve knowledge transfer
4. **Progressive distillation** works well for complex tasks
5. **Architecture design** significantly impacts distillation success

Proper implementation can achieve significant model compression while maintaining or improving performance. 