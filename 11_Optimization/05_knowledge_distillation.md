# Knowledge Distillation

Knowledge distillation is a technique for transferring knowledge from a large, complex model (teacher) to a smaller, simpler model (student) while maintaining or improving performance.

> **Key Insight:** Knowledge distillation enables you to deploy efficient models on edge devices without sacrificing much accuracy.

> **Did you know?** Many mobile AI applications (e.g., voice assistants, image recognition) use knowledge distillation to shrink large models for real-time use!

## Overview

Knowledge distillation enables:
- **Model compression** for deployment on resource-constrained devices
- **Performance improvement** through teacher guidance
- **Knowledge transfer** from ensemble models
- **Privacy preservation** by training on soft targets

> **Geometric Intuition:** Imagine a student learning from a teacher. The teacher not only gives the right answers (hard labels) but also explains why (soft probabilities), helping the student generalize better.

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

> **Common Pitfall:** If the temperature $`T`$ is too low, soft targets become too similar to hard labels; if too high, they become too uniform and lose useful information.

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

> **Key Insight:** The KL divergence term encourages the student to match the teacher's output distribution, not just the correct class.

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
```
*This trainer combines hard and soft losses to guide the student model using both ground-truth labels and teacher predictions.*

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
```
*Feature matching encourages the student to mimic not just the teacher's outputs, but also its internal representations.*

---

> **Try it yourself!** Experiment with different temperatures and alpha values. How do they affect the student's learning and final accuracy?

> **Key Insight:** Knowledge distillation is a powerful tool for model compression, transfer learning, and privacy-preserving AI. 