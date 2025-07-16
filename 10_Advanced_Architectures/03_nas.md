# Neural Architecture Search (NAS)

Neural Architecture Search (NAS) is a technique for automating the design of neural network architectures. Instead of manually crafting models, NAS algorithms search for the best architecture for a given task and dataset.

## 1. Motivation

Designing neural network architectures is time-consuming and requires expertise. NAS aims to:
- Discover high-performing architectures automatically
- Reduce human bias in model design
- Adapt architectures to specific tasks or hardware constraints

## 2. NAS Components

A typical NAS framework consists of:
- **Search Space:** Defines possible architectures (e.g., layer types, connections, hyperparameters)
- **Search Strategy:** How to explore the space (e.g., reinforcement learning, evolutionary algorithms, random search)
- **Performance Estimation:** How to evaluate candidate architectures (e.g., full training, proxy tasks, early stopping)

## 3. NAS Objective

The goal is to find the architecture $`a^*`$ that maximizes performance:
```math
\text{NAS Objective:} \quad a^* = \underset{a \in \mathcal{A}}{\mathrm{argmax}}\; \text{Accuracy}(a)
```
where $`\mathcal{A}`$ is the set of possible architectures.

## 4. Search Strategies

### 4.1 Reinforcement Learning (RL)-based NAS
A controller (often an RNN) generates architectures. The performance of each architecture is used as a reward to update the controller.

**RL Update:**
```math
\theta \leftarrow \theta + \alpha \nabla_\theta \mathbb{E}_{a \sim \pi_\theta}[R(a)]
```
where $`\pi_\theta`$ is the controller policy and $`R(a)`$ is the reward (e.g., validation accuracy).

### 4.2 Evolutionary Algorithms
A population of architectures is evolved over generations using mutation and crossover. The best-performing architectures are selected for the next generation.

### 4.3 Random Search
Randomly samples architectures from the search space. Surprisingly effective for some tasks.

## 5. Example: Random Search NAS in Python

Below is a simple example of random search for NAS using PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define a simple search space
search_space = [
    {'hidden_size': 16, 'activation': nn.ReLU},
    {'hidden_size': 32, 'activation': nn.Tanh},
    {'hidden_size': 64, 'activation': nn.ELU},
]

def build_model(input_dim, output_dim, config):
    return nn.Sequential(
        nn.Linear(input_dim, config['hidden_size']),
        config['activation'](),
        nn.Linear(config['hidden_size'], output_dim)
    )

# Dummy data
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

best_acc = 0
best_model = None
for config in search_space:
    model = build_model(10, 2, config)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    # Train for a few epochs (proxy)
    for epoch in range(5):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
    # Evaluate
    acc = (output.argmax(dim=1) == y).float().mean().item()
    print(f"Config: {config}, Accuracy: {acc:.2f}")
    if acc > best_acc:
        best_acc = acc
        best_model = model

print("Best accuracy:", best_acc)
```

## 6. Challenges and Trends
- **Search cost:** NAS can be computationally expensive
- **Transferability:** Architectures found on one dataset may not generalize
- **Hardware-aware NAS:** Optimizing for speed, memory, or energy

## 7. Further Reading
- [Zoph & Le, 2017: Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)
- [Elsken et al., 2019: Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377) 