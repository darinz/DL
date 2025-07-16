# Neural Architecture Search (NAS)

Neural Architecture Search (NAS) is a technique for automating the design of neural network architectures. Instead of manually crafting models, NAS algorithms search for the best architecture for a given task and dataset.

> **Explanation:**
> NAS is like having an automated scientist that tries out many different neural network designs and picks the best one for your problem. This can lead to better models than those designed by hand.

> **Key Insight:** NAS automates model design, potentially discovering novel architectures that outperform human-designed ones.

> **Did you know?** Some state-of-the-art models for image classification and NLP were discovered using NAS!

## 1. Motivation

Designing neural network architectures is time-consuming and requires expertise. NAS aims to:
- Discover high-performing architectures automatically
- Reduce human bias in model design
- Adapt architectures to specific tasks or hardware constraints

> **Geometric Intuition:** Imagine searching a vast landscape of possible neural networks. NAS algorithms are like explorers, seeking the highest peaks (best models) in this landscape.

## 2. NAS Components

A typical NAS framework consists of:
- **Search Space:** Defines possible architectures (e.g., layer types, connections, hyperparameters)
- **Search Strategy:** How to explore the space (e.g., reinforcement learning, evolutionary algorithms, random search)
- **Performance Estimation:** How to evaluate candidate architectures (e.g., full training, proxy tasks, early stopping)

> **Explanation:**
> The search space is like the menu of possible model parts. The search strategy is the method for picking which models to try. Performance estimation is how you decide which models are good.

> **Common Pitfall:** A too-large search space can make NAS intractable; a too-small space may miss the best architectures.

## 3. NAS Objective

The goal is to find the architecture $`a^*`$ that maximizes performance:
```math
\text{NAS Objective:} \quad a^* = \underset{a \in \mathcal{A}}{\mathrm{argmax}}\; \text{Accuracy}(a)
```
> **Math Breakdown:**
> - $`\mathcal{A}`$: The set of all possible architectures.
> - $`a^*`$: The best architecture found by the search.
> - $`\mathrm{argmax}`$: The operation that finds the $`a`$ with the highest accuracy.
> - $`\text{Accuracy}(a)`$: The performance metric (could be accuracy, speed, etc.).

### Step-by-Step Breakdown
1. **Define the search space** $`\mathcal{A}`$ (e.g., possible layers, connections).
   > **Explanation:**
   > Decide what kinds of models the search is allowed to try.
2. **Choose a search strategy** (RL, evolution, random, etc.).
   > **Explanation:**
   > Pick a method for exploring the search space.
3. **Sample candidate architectures** $`a`$ from $`\mathcal{A}`$.
   > **Explanation:**
   > Try out different models from the search space.
4. **Estimate performance** (e.g., train and validate).
   > **Explanation:**
   > Train each candidate model and see how well it does.
5. **Update the search strategy** based on results.
   > **Explanation:**
   > Use the results to guide future choices (e.g., reward good models).
6. **Repeat** until a satisfactory architecture is found.
   > **Explanation:**
   > Keep searching until you find a model that meets your needs.

## 4. Search Strategies

### 4.1 Reinforcement Learning (RL)-based NAS
A controller (often an RNN) generates architectures. The performance of each architecture is used as a reward to update the controller.

**RL Update:**
```math
\theta \leftarrow \theta + \alpha \nabla_\theta \mathbb{E}_{a \sim \pi_\theta}[R(a)]
```
> **Math Breakdown:**
> - $`\theta`$: Parameters of the controller (e.g., RNN weights).
> - $`\alpha`$: Learning rate.
> - $`\pi_\theta`$: Policy for generating architectures.
> - $`R(a)`$: Reward for architecture $`a`$ (e.g., validation accuracy).
> - The controller is updated to generate better architectures over time.

> **Key Insight:** RL-based NAS can learn to generate better architectures over time, but can be slow and resource-intensive.

### 4.2 Evolutionary Algorithms
A population of architectures is evolved over generations using mutation and crossover. The best-performing architectures are selected for the next generation.

> **Explanation:**
> Evolutionary NAS mimics natural selection: models "reproduce" with variation, and the best ones survive.

> **Did you know?** Evolutionary NAS can discover diverse and robust architectures, sometimes outperforming RL-based methods.

### 4.3 Random Search
Randomly samples architectures from the search space. Surprisingly effective for some tasks.

> **Explanation:**
> Even just trying random models can sometimes find good solutions, especially if the search space is well-designed.

> **Try it yourself!** Implement a random search NAS for a small dataset. How does it compare to hand-designed models?

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
> **Code Walkthrough:**
> - Defines a small search space of possible model configurations (hidden size and activation).
> - For each configuration, builds a model, trains it briefly, and evaluates its accuracy.
> - Keeps track of the best model found.
> - This is a simple example of NAS using random search.

*This code samples different architectures, trains them briefly, and selects the best one based on validation accuracy.*

## 6. Challenges and Trends
- **Search cost:** NAS can be computationally expensive
- **Transferability:** Architectures found on one dataset may not generalize
- **Hardware-aware NAS:** Optimizing for speed, memory, or energy

> **Key Insight:** Modern NAS research focuses on making search more efficient and architectures more transferable and hardware-friendly.

## 7. Further Reading
- [Zoph & Le, 2017: Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)
- [Elsken et al., 2019: Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377) 