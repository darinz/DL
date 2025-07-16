# Meta-Learning (Learning to Learn)

Meta-learning, or "learning to learn," is a paradigm where models are trained to quickly adapt to new tasks with minimal data. This is especially useful in few-shot learning scenarios.

## 1. Motivation

Traditional deep learning models require large amounts of data and training time for each new task. Meta-learning aims to:
- Enable fast adaptation to new tasks
- Leverage experience from previous tasks
- Improve generalization with few examples

## 2. Meta-Learning Framework

Meta-learning typically involves two loops:
- **Inner loop:** Learns a task-specific model using a small dataset
- **Outer loop:** Updates meta-parameters to improve adaptation across tasks

## 3. Model-Agnostic Meta-Learning (MAML)

MAML seeks initial parameters $`\theta`$ such that a few gradient steps on a new task yield good performance.

### MAML Algorithm
1. Sample a batch of tasks $`\mathcal{T}_i`$ from a task distribution $`p(\mathcal{T})`$
2. For each task, compute adapted parameters:
   $`
   \theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)
   `$
3. Update meta-parameters using the adapted parameters:
   ```math
   \theta \leftarrow \theta - \beta \nabla_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(\theta'_i)
   ```
where $`\alpha`$ and $`\beta`$ are learning rates.

### MAML Example (Python)
Below is a simplified MAML implementation for regression using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
    def forward(self, x):
        return self.fc(x)

def maml_step(model, loss_fn, x_train, y_train, x_val, y_val, alpha, beta):
    # Clone model for inner loop
    fast_weights = list(model.parameters())
    # Inner loop
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
    fast_weights = [w - alpha * g for w, g in zip(fast_weights, grads)]
    # Outer loop
    y_pred_val = model(x_val)
    loss_val = loss_fn(y_pred_val, y_val)
    loss_val.backward()
    # Update meta-parameters
    for param in model.parameters():
        param.data -= beta * param.grad
    model.zero_grad()
```

## 4. Reptile

Reptile is a simpler meta-learning algorithm that also learns initial parameters for fast adaptation, but uses repeated sampling and gradient steps across tasks.

### Reptile Algorithm
1. Sample a task $`\mathcal{T}`$
2. Train on $`\mathcal{T}`$ for $`k`$ steps to get $`\theta'`$
3. Update meta-parameters:
   ```math
   \theta \leftarrow \theta + \epsilon (\theta' - \theta)
   ```
where $`\epsilon`$ is the meta step size.

### Reptile Example (Python)

```python
# Assume model, optimizer, and data are defined
meta_lr = 0.1
for task in tasks:
    # Save initial parameters
    theta_init = [p.clone() for p in model.parameters()]
    # Train on task
    for step in range(k):
        # ... standard training step ...
        pass
    # Update meta-parameters
    for p, p_init in zip(model.parameters(), theta_init):
        p.data = p_init.data + meta_lr * (p.data - p_init.data)
```

## 5. Applications
- Few-shot image classification
- Reinforcement learning
- Robotics

## 6. Further Reading
- [Finn et al., 2017: Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [Nichol et al., 2018: On First-Order Meta-Learning Algorithms](https://arxiv.org/abs/1803.02999) 