# Learning Rate Strategies in Deep Learning

> **Key Insight:** The learning rate is the most important hyperparameter in deep learning. The right learning rate schedule can dramatically improve convergence, stability, and final model performance.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Fixed Learning Rate](#fixed-learning-rate)
3. [Step Decay](#step-decay)
4. [Exponential Decay](#exponential-decay)
5. [Cosine Annealing](#cosine-annealing)
6. [Warmup Strategies](#warmup-strategies)
7. [Cyclic Learning Rates](#cyclic-learning-rates)
8. [Implementation in Python](#implementation-in-python)
9. [Learning Rate Finder](#learning-rate-finder)
10. [Practical Guidelines](#practical-guidelines)
11. [Summary](#summary)

---

## Introduction

> **Intuition:** The learning rate is like the gas pedal in a car—press too hard and you might crash (diverge), too soft and you'll never reach your destination (slow learning).

### What is Learning Rate Scheduling?

Learning rate scheduling involves adapting the learning rate $`\eta_t`$ during training based on the current step $`t`$ or epoch. The goal is to:

1. **Start Fast:** Use high learning rates for rapid initial progress
2. **Converge Precisely:** Use low learning rates for fine-tuning
3. **Escape Local Minima:** Use varying rates to explore the loss landscape
4. **Maintain Stability:** Prevent training divergence

### Mathematical Framework

The general form of learning rate scheduling is:

```math
\eta_t = \eta_0 \cdot f(t, \text{parameters})
```

Where:
- $`\eta_0`$ is the initial learning rate
- $`f(t, \text{parameters})`$ is the scheduling function
- $`t`$ is the current training step or epoch

### Key Considerations

1. **Problem Type:** Different problems require different schedules
2. **Model Size:** Larger models often need different strategies
3. **Dataset Size:** Affects the optimal schedule length
4. **Optimizer Choice:** Some optimizers have built-in scheduling

> **Did you know?**
> The learning rate is sometimes called the "step size" in optimization literature. In deep learning, "learning rate" is the standard term.

> **Annotation:** The right learning rate schedule can make the difference between a model that converges quickly and one that never learns at all.

---

## Fixed Learning Rate

> **Annotation:** Fixed learning rates are simple but rarely optimal. Most modern training uses some form of scheduling.

### Basic Fixed Learning Rate

**Formula:**
$`\eta_t = \eta_0`$

- **Intuition:** Keeps the learning rate constant throughout training. Simple but often suboptimal.
- **Properties:** Requires careful tuning, can lead to suboptimal convergence, good baseline for comparison.

### Implementation

```python
class FixedLR:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
    
    def get_lr(self, step):
        return self.learning_rate
```

> **Common Pitfall:**
> Using a fixed learning rate can cause the model to get stuck or oscillate if the rate is not well-tuned. Always monitor training loss for signs of instability.

---

## Step Decay

> **Intuition:** Step decay mimics the way humans learn—make big changes at first, then smaller adjustments as you get closer to the goal.

### Step Decay Formula

**Formula:**
$`\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}`$

Where:
- $`\gamma`$ is the decay factor (typically 0.1 or 0.5)
- $`s`$ is the step size (number of steps between decays)
- $`\lfloor t/s \rfloor`$ is the floor division

- **Intuition:** Reduces the learning rate by a fixed factor every $`s`$ steps. Mimics manual learning rate drops.
- **Properties:** Simple, effective, clear decay pattern, hyperparameters $`\gamma`$ and $`s`$ need tuning.

### Example Schedules

1. **Aggressive Decay:** $`\gamma = 0.1, s = 30`$
2. **Moderate Decay:** $`\gamma = 0.5, s = 50`$
3. **Slow Decay:** $`\gamma = 0.8, s = 100`$

### Implementation

```python
class StepDecay:
    def __init__(self, initial_lr=0.001, decay_factor=0.1, step_size=30):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size
    
    def get_lr(self, step):
        return self.initial_lr * (self.decay_factor ** (step // self.step_size))
```

> **Try it yourself!**
> Train a model with and without step decay. Compare the training and validation loss curves.

---

## Exponential Decay

> **Annotation:** Exponential decay provides a smooth, continuous reduction in learning rate, which can help with fine-tuning and stability.

### Exponential Decay Formula

**Formula:**
$`\eta_t = \eta_0 \cdot e^{-\lambda t}`$

Where $`\lambda`$ is the decay rate.

- **Intuition:** Smoothly decreases the learning rate over time. Good for gradual fine-tuning.
- **Properties:** Continuous decay, rapid early decay, very small learning rates later, $`\lambda`$ controls decay speed.

### Alternative Forms

1. **Time-based Decay:**
$`\eta_t = \frac{\eta_0}{1 + \lambda t}`$
2. **Inverse Time Decay:**
$`\eta_t = \frac{\eta_0}{1 + \lambda \sqrt{t}}`$

### Implementation

```python
class ExponentialDecay:
    def __init__(self, initial_lr=0.001, decay_rate=0.01):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def get_lr(self, step):
        return self.initial_lr * np.exp(-self.decay_rate * step)

class TimeDecay:
    def __init__(self, initial_lr=0.001, decay_rate=0.01):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def get_lr(self, step):
        return self.initial_lr / (1 + self.decay_rate * step)
```

> **Common Pitfall:**
> If the decay rate is too high, the learning rate can become too small too quickly, causing training to stall.

---

## Cosine Annealing

> **Intuition:** Cosine annealing lets the learning rate rise and fall in cycles, which can help the optimizer escape local minima and explore the loss landscape.

### Cosine Annealing Formula

**Formula:**
$`\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))`$

Where:
- $`\eta_{max}`$ is the maximum learning rate
- $`\eta_{min}`$ is the minimum learning rate
- $`T`$ is the total number of steps

- **Intuition:** Smoothly decreases the learning rate following a cosine curve. Can be restarted for multiple cycles.
- **Properties:** Often outperforms step decay, hyperparameters $`\eta_{max}`$, $`\eta_{min}`$, $`T`$.

### Cosine Annealing with Restarts

**Formula:**
$`\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t \bmod T_i}{T_i}\pi))`$

Where $`T_i`$ is the restart period for cycle $`i`$.

### Implementation

```python
class CosineAnnealing:
    def __init__(self, max_lr=0.001, min_lr=1e-6, T_max=1000):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_max = T_max
    
    def get_lr(self, step):
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
               (1 + np.cos(np.pi * step / self.T_max))

class CosineAnnealingWarmRestarts:
    def __init__(self, max_lr=0.001, min_lr=1e-6, T_0=100, T_mult=2):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_0 = T_0
        self.T_mult = T_mult
    
    def get_lr(self, step):
        # Find current cycle
        cycle = 0
        T_curr = self.T_0
        T_sum = 0
        
        while T_sum + T_curr <= step:
            T_sum += T_curr
            T_curr *= self.T_mult
            cycle += 1
        
        # Current position in cycle
        t_curr = step - T_sum
        
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
               (1 + np.cos(np.pi * t_curr / T_curr))
```

> **Did you know?**
> Cosine annealing was popularized by the SGDR paper and is now widely used in training large models, especially in computer vision and NLP.

---

## Warmup Strategies

> **Annotation:** Warmup helps prevent instability at the start of training, especially for large models or high learning rates.

### Why Warmup?

Warmup is crucial for:
- **Large Models:** Prevent early instability
- **High Learning Rates:** Gradual adaptation
- **Batch Normalization:** Allow statistics to stabilize
- **Transformer Models:** Essential for training stability

### Linear Warmup

**Formula:**
$`\eta_t = \eta_{max} \cdot \frac{t}{T_{warmup}}`$

Where $`T_{warmup}`$ is the warmup period.

### Cosine Warmup

**Formula:**
$`\eta_t = \eta_{max} \cdot \frac{1}{2}(1 + \cos(\pi - \frac{t}{T_{warmup}}\pi))`$

### Implementation

```python
class LinearWarmup:
    def __init__(self, max_lr=0.001, warmup_steps=1000):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        return self.max_lr

class CosineWarmup:
    def __init__(self, max_lr=0.001, warmup_steps=1000):
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * 0.5 * (1 + np.cos(np.pi - np.pi * step / self.warmup_steps))
        return self.max_lr
```

> **Common Pitfall:**
> Not using warmup with large models or high learning rates can cause training to diverge in the early epochs.

---

## Cyclic Learning Rates

### One Cycle Policy

**Formula:**
$`\eta_t = \begin{cases}
\eta_{max} \cdot \frac{t}{T_{warmup}} & \text{if } t < T_{warmup} \\
\eta_{max} \cdot (1 - \frac{t - T_{warmup}}{T_{anneal}}) & \text{if } T_{warmup} \leq t < T_{warmup} + T_{anneal} \\
\eta_{min} & \text{otherwise}
\end{cases}`$

- **Intuition:** Increases the learning rate, then decreases it, often leading to faster convergence and better generalization.
- **Properties:** Rapid convergence, good generalization, hyperparameters $`\eta_{max}`$, $`T_{warmup}`$, $`T_{anneal}`$.

### Implementation

```python
class OneCycleLR:
    def __init__(self, max_lr=0.001, min_lr=1e-6, total_steps=1000, 
                 warmup_ratio=0.3, anneal_ratio=0.7):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.anneal_steps = int(total_steps * anneal_ratio)
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            # Warmup phase
            return self.max_lr * step / self.warmup_steps
        elif step < self.warmup_steps + self.anneal_steps:
            # Annealing phase
            t = step - self.warmup_steps
            return self.max_lr * (1 - t / self.anneal_steps)
        else:
            # Final phase
            return self.min_lr
```

> **Try it yourself!**
> Plot the learning rate schedule for a one cycle policy. How does it compare to step decay or cosine annealing?

---

## Implementation in Python

### Complete Learning Rate Scheduler Framework

```python
import numpy as np
import matplotlib.pyplot as plt

class LearningRateScheduler:
    """Base class for learning rate schedulers"""
    
    def __init__(self, initial_lr=0.001):
        self.initial_lr = initial_lr
        self.current_step = 0
    
    def get_lr(self, step):
        """Get learning rate for current step"""
        raise NotImplementedError
    
    def step(self):
        """Advance scheduler by one step"""
        lr = self.get_lr(self.current_step)
        self.current_step += 1
        return lr
    
    def plot_schedule(self, total_steps=1000):
        """Plot the learning rate schedule"""
        steps = np.arange(total_steps)
        lrs = [self.get_lr(step) for step in steps]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lrs)
        plt.xlabel('Training Step')
        plt.ylabel('Learning Rate')
        plt.title(f'{self.__class__.__name__} Schedule')
        plt.yscale('log')
        plt.grid(True)
        plt.show()

class StepDecayScheduler(LearningRateScheduler):
    def __init__(self, initial_lr=0.001, decay_factor=0.1, step_size=100):
        super().__init__(initial_lr)
        self.decay_factor = decay_factor
        self.step_size = step_size
    
    def get_lr(self, step):
        return self.initial_lr * (self.decay_factor ** (step // self.step_size))

class ExponentialDecayScheduler(LearningRateScheduler):
    def __init__(self, initial_lr=0.001, decay_rate=0.001):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
    
    def get_lr(self, step):
        return self.initial_lr * np.exp(-self.decay_rate * step)

class CosineAnnealingScheduler(LearningRateScheduler):
    def __init__(self, max_lr=0.001, min_lr=1e-6, T_max=1000):
        super().__init__(max_lr)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.T_max = T_max
    
    def get_lr(self, step):
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
               (1 + np.cos(np.pi * step / self.T_max))

class WarmupCosineScheduler(LearningRateScheduler):
    def __init__(self, max_lr=0.001, min_lr=1e-6, warmup_steps=100, total_steps=1000):
        super().__init__(max_lr)
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * step / self.warmup_steps
        else:
            # Cosine annealing
            t = step - self.warmup_steps
            T = self.total_steps - self.warmup_steps
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * \
                   (1 + np.cos(np.pi * t / T))

# Example usage
if __name__ == "__main__":
    # Create different schedulers
    schedulers = {
        'Step Decay': StepDecayScheduler(initial_lr=0.001, decay_factor=0.1, step_size=200),
        'Exponential Decay': ExponentialDecayScheduler(initial_lr=0.001, decay_rate=0.001),
        'Cosine Annealing': CosineAnnealingScheduler(max_lr=0.001, min_lr=1e-6, T_max=1000),
        'Warmup + Cosine': WarmupCosineScheduler(max_lr=0.001, min_lr=1e-6, warmup_steps=100, total_steps=1000)
    }
    
    # Plot all schedules
    plt.figure(figsize=(12, 8))
    for name, scheduler in schedulers.items():
        steps = np.arange(1000)
        lrs = [scheduler.get_lr(step) for step in steps]
        plt.plot(steps, lrs, label=name)
    
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules Comparison')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()
```

> **Code Commentary:**
> - Each scheduler is implemented as a class for modularity and extensibility.
> - The `plot_schedule` method helps visualize how the learning rate changes over time.

### Integration with Optimizers

```python
class ScheduledOptimizer:
    """Optimizer with learning rate scheduling"""
    
    def __init__(self, optimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def step(self, params, grads):
        """Update parameters with scheduled learning rate"""
        # Update learning rate
        lr = self.scheduler.step()
        self.optimizer.learning_rate = lr
        
        # Update parameters
        self.optimizer.step(params, grads)
    
    def get_lr(self):
        """Get current learning rate"""
        return self.optimizer.learning_rate

# Example training with scheduler
def train_with_scheduler(model, train_data, scheduler_type='cosine', epochs=100):
    """Train model with specified learning rate scheduler"""
    
    # Create optimizer
    optimizer = Adam(learning_rate=0.001)
    
    # Create scheduler
    if scheduler_type == 'step':
        scheduler = StepDecayScheduler(initial_lr=0.001, decay_factor=0.1, step_size=30)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingScheduler(max_lr=0.001, min_lr=1e-6, T_max=epochs)
    elif scheduler_type == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(max_lr=0.001, min_lr=1e-6, 
                                         warmup_steps=10, total_steps=epochs)
```

> **Try it yourself!**
> Train a model with different learning rate schedules and compare their convergence speed and final accuracy.

---

## Learning Rate Finder

### Automatic Learning Rate Discovery

The learning rate finder automatically discovers the optimal learning rate range:

```python
class LearningRateFinder:
    def __init__(self, model, optimizer, loss_fn, min_lr=1e-7, max_lr=1, num_steps=100):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
    
    def find_lr(self, train_data):
        """Find optimal learning rate range"""
        # Exponential growth factor
        growth_factor = (self.max_lr / self.min_lr) ** (1 / self.num_steps)
        
        # Store original learning rate
        original_lr = self.optimizer.learning_rate
        
        lrs = []
        losses = []
        
        for step in range(self.num_steps):
            # Update learning rate
            lr = self.min_lr * (growth_factor ** step)
            self.optimizer.learning_rate = lr
            
            # Training step
            loss = self.train_step(train_data)
            
            lrs.append(lr)
            losses.append(loss)
            
            # Stop if loss explodes
            if step > 0 and losses[-1] > 4 * losses[0]:
                break
        
        # Restore original learning rate
        self.optimizer.learning_rate = original_lr
        
        return lrs, losses
    
    def plot_lr_finder(self, lrs, losses):
        """Plot learning rate finder results"""
        plt.figure(figsize=(10, 6))
        plt.semilogx(lrs, losses)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.grid(True)
        plt.show()
        
        # Find optimal learning rate (minimum loss)
        optimal_idx = np.argmin(losses)
        optimal_lr = lrs[optimal_idx]
        print(f"Optimal learning rate: {optimal_lr:.2e}")
        
        return optimal_lr
    
    def train_step(self, train_data):
        """Single training step"""
        # Forward pass
        x, y = train_data
        y_pred = self.model.forward(x)
        
        # Compute loss
        loss = self.loss_fn.compute(y, y_pred)
        
        # Backward pass
        grads = self.model.backward(x, y)
        
        # Update parameters
        self.optimizer.step(self.model.parameters(), grads)
        
        return loss

# Example usage
if __name__ == "__main__":
    # Create model and optimizer
    model = SimpleNeuralNetwork([2, 4, 1])
    optimizer = Adam(learning_rate=0.001)
    loss_fn = MeanSquaredError()
    
    # Create learning rate finder
    lr_finder = LearningRateFinder(model, optimizer, loss_fn)
    
    # Generate some training data
    X = np.random.randn(2, 100)
    y = np.random.randn(1, 100)
    train_data = (X, y)
    
    # Find optimal learning rate
    lrs, losses = lr_finder.find_lr(train_data)
    optimal_lr = lr_finder.plot_lr_finder(lrs, losses)
```

> **Did you know?**
> The learning rate finder technique was popularized by the fastai library and is now a standard tool for tuning deep learning models.

---

## Practical Guidelines

### Scheduler Selection Guidelines

| Problem Type | Recommended Scheduler | Reasoning |
|--------------|----------------------|-----------|
| Small models | Fixed LR or Step Decay | Simple, effective |
| Large models | Warmup + Cosine Annealing | Stability, good convergence |
| Transformers | Warmup + Linear Decay | Proven effective |
| Computer Vision | Step Decay or Cosine Annealing | Standard practice |
| NLP | Warmup + Cosine Annealing | Good for language models |

### Hyperparameter Tuning

```python
def tune_learning_rate_schedule(model_class, train_data, val_data, 
                               scheduler_types=['step', 'cosine', 'warmup_cosine']):
    """Tune learning rate schedule hyperparameters"""
    best_score = float('inf')
    best_config = {}
    
    for scheduler_type in scheduler_types:
        if scheduler_type == 'step':
            # Tune step decay parameters
            for decay_factor in [0.1, 0.5, 0.8]:
                for step_size in [30, 50, 100]:
                    scheduler = StepDecayScheduler(initial_lr=0.001, 
                                                 decay_factor=decay_factor, 
                                                 step_size=step_size)
                    score = evaluate_schedule(model_class, train_data, val_data, scheduler)
                    
                    if score < best_score:
                        best_score = score
                        best_config = {
                            'type': scheduler_type,
                            'decay_factor': decay_factor,
                            'step_size': step_size
                        }
        
        elif scheduler_type == 'cosine':
            # Tune cosine annealing parameters
            for T_max in [500, 1000, 2000]:
                scheduler = CosineAnnealingScheduler(max_lr=0.001, min_lr=1e-6, T_max=T_max)
                score = evaluate_schedule(model_class, train_data, val_data, scheduler)
                
                if score < best_score:
                    best_score = score
                    best_config = {
                        'type': scheduler_type,
                        'T_max': T_max
                    }
    
    return best_config, best_score

def evaluate_schedule(model_class, train_data, val_data, scheduler, epochs=50):
    """Evaluate a learning rate schedule"""
    model = model_class()
    optimizer = Adam(learning_rate=0.001)
    scheduled_opt = ScheduledOptimizer(optimizer, scheduler)
    
    # Train model
    losses = train_with_scheduler(model, train_data, scheduled_opt, epochs)
    
    # Evaluate on validation set
    val_score = evaluate_model(model, val_data)
    
    return val_score
```

### Monitoring and Debugging

```python
class LearningRateMonitor:
    """Monitor learning rate during training"""
    
    def __init__(self):
        self.lr_history = []
        self.loss_history = []
    
    def log(self, step, lr, loss):
        """Log learning rate and loss"""
        self.lr_history.append((step, lr))
        self.loss_history.append((step, loss))
    
    def plot_training_curves(self):
        """Plot learning rate and loss curves"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Learning rate curve
        steps, lrs = zip(*self.lr_history)
        ax1.semilogy(steps, lrs)
        ax1.set_ylabel('Learning Rate')
        ax1.set_title('Learning Rate Schedule')
        ax1.grid(True)
        
        # Loss curve
        steps, losses = zip(*self.loss_history)
        ax2.plot(steps, losses)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_convergence(self):
        """Analyze training convergence"""
        steps, losses = zip(*self.loss_history)
        
        # Check for convergence
        if len(losses) > 100:
            recent_losses = losses[-100:]
            if np.std(recent_losses) < 1e-6:
                print("Training has converged")
            elif np.mean(recent_losses) > np.mean(losses[:100]):
                print("Training may be diverging")
            else:
                print("Training is progressing normally")

# Usage in training
def train_with_monitoring(model, train_data, scheduler, epochs=100):
```

> **Try it yourself!**
> Use the `LearningRateMonitor` to track and visualize the learning rate and loss during training. Can you spot signs of overfitting or divergence?

---

## Summary

Learning rate strategies are essential for effective deep learning training:

- $`\textbf{Fixed}`$: Simple but rarely optimal
- $`\textbf{Step/Exponential Decay}`$: Good for staged or gradual convergence
- $`\textbf{Cosine Annealing}`$: Smooth, often best for large models
- $`\textbf{Warmup}`$: Crucial for stability in large/modern architectures
- $`\textbf{Cyclic/One Cycle}`$: Fast convergence and good generalization

Key considerations:
- **Problem Type:** Different schedules work better for different problems
- **Model/Dataset Size:** Larger models and datasets often need more sophisticated schedules
- **Monitoring:** Always track learning rate and loss during training
- **Tuning:** Use learning rate finders and monitoring tools to optimize your schedule

> **Key Insight:**
> Mastering learning rate schedules is one of the most effective ways to boost your deep learning results. Experiment, monitor, and tune for best performance! 