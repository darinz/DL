# Federated Learning

Federated learning is a machine learning approach that enables training models on decentralized data without sharing raw data, preserving privacy while leveraging distributed datasets.

> **Key Insight:** Federated learning allows collaborative model training across many devices or organizations, all while keeping sensitive data local and private.

> **Did you know?** Federated learning is used in real-world applications like keyboard prediction on smartphones, where user data never leaves the device!

## Overview

Federated learning addresses privacy concerns by:
- **Keeping data local** on client devices
- **Sharing only model updates** instead of raw data
- **Enabling collaborative training** across multiple parties
- **Preserving data privacy** and regulatory compliance

> **Geometric Intuition:** Imagine each device as a chef perfecting a recipe with local ingredients. Instead of sharing the ingredients, each chef shares their improved recipe, which is then averaged to create a better global recipe.

## Mathematical Foundation

### Federated Averaging (FedAvg)

The core algorithm aggregates local model updates:

```math
w_{t+1} = \sum_{k=1}^{K} \frac{n_k}{n} w_t^{(k)}
```

where:
- $`w_{t+1}`$ = global model at round $`t+1`$
- $`w_t^{(k)}`$ = local model of client $`k`$ at round $`t`$
- $`n_k`$ = number of samples on client $`k`$
- $`n`$ = total number of samples across all clients

### Local Training

Each client performs local SGD:
```math
w_{t+1}^{(k)} = w_t^{(k)} - \alpha \nabla L_k(w_t^{(k)})
```

### Convergence Analysis

For FedAvg with $`K`$ clients and $`T`$ rounds:
```math
\mathbb{E}[f(\bar{w}_T)] - f(w^*) \leq O\left(\frac{1}{\sqrt{T}} + \frac{1}{\sqrt{K}}\right)
```

> **Common Pitfall:** If client data is highly non-i.i.d. (not identically distributed), convergence can be slow or unstable. Techniques like FedProx or personalization can help.

## Implementation Strategies

### 1. Basic Federated Learning

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict

class FederatedTrainer:
    def __init__(self, global_model, num_clients=10):
        self.global_model = global_model
        self.num_clients = num_clients
        self.client_models = []
        
        # Initialize client models
        for _ in range(num_clients):
            client_model = type(global_model)()
            client_model.load_state_dict(global_model.state_dict())
            self.client_models.append(client_model)
    
    def train_client(self, client_id, data_loader, epochs=1, lr=0.001):
        """Train a single client model."""
        model = self.client_models[client_id]
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model.state_dict()
    
    def federated_averaging(self, client_weights, client_sizes):
        """Perform federated averaging."""
        total_samples = sum(client_sizes)
        
        # Initialize averaged weights
        averaged_weights = OrderedDict()
        
        # Get parameter names from first client
        param_names = client_weights[0].keys()
        
        for param_name in param_names:
            # Weighted average of parameters
            averaged_param = torch.zeros_like(client_weights[0][param_name])
            
            for client_id in range(len(client_weights)):
                weight = client_sizes[client_id] / total_samples
                averaged_param += weight * client_weights[client_id][param_name]
            
            averaged_weights[param_name] = averaged_param
        
        return averaged_weights
    
    def train_round(self, client_data_loaders, client_sizes):
        """Train one round of federated learning."""
        client_weights = []
        
        # Train each client
        for client_id in range(self.num_clients):
            if client_id < len(client_data_loaders):
                weights = self.train_client(client_id, client_data_loaders[client_id])
                client_weights.append(weights)
            else:
                # Use current global model weights for inactive clients
                client_weights.append(self.global_model.state_dict())
        
        # Federated averaging
        averaged_weights = self.federated_averaging(client_weights, client_sizes)
        
        # Update global model
        self.global_model.load_state_dict(averaged_weights)
        
        # Update client models
        for client_model in self.client_models:
            client_model.load_state_dict(averaged_weights)
        
        return averaged_weights
```
*This trainer simulates federated learning by training local models and averaging their weights.*

### 2. Advanced Federated Learning with FedProx

```python
class FedProxTrainer:
    def __init__(self, global_model, num_clients=10, mu=0.01):
        self.global_model = global_model
        self.num_clients = num_clients
        self.mu = mu  # Proximal term coefficient
        self.client_models = []
        
        # Initialize client models
        for _ in range(num_clients):
            client_model = type(global_model)()
            client_model.load_state_dict(global_model.state_dict())
            self.client_models.append(client_model)
    
    def proximal_term(self, local_model, global_model):
        """Compute proximal term for FedProx."""
        proximal_loss = 0
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            proximal_loss += torch.norm(local_param - global_param) ** 2
        return 0.5 * self.mu * proximal_loss
    
    def train_client_fedprox(self, client_id, data_loader, epochs=1, lr=0.001):
        """Train client with FedProx."""
        model = self.client_models[client_id]
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                
                # Standard loss
                output = model(data)
                loss = criterion(output, target)
                
                # Add proximal term
                proximal_loss = self.proximal_term(model, self.global_model)
                total_loss = loss + proximal_loss
                
                total_loss.backward()
                optimizer.step()
        
        return model.state_dict()
    
    def train_round_fedprox(self, client_data_loaders, client_sizes):
        """Train one round with FedProx."""
        client_weights = []
        
        # Train each client with FedProx
        for client_id in range(self.num_clients):
            if client_id < len(client_data_loaders):
                weights = self.train_client_fedprox(client_id, client_data_loaders[client_id])
                client_weights.append(weights)
            else:
                client_weights.append(self.global_model.state_dict())
        
        # Federated averaging
        averaged_weights = self.federated_averaging(client_weights, client_sizes)
        
        # Update models
        self.global_model.load_state_dict(averaged_weights)
        for client_model in self.client_models:
            client_model.load_state_dict(averaged_weights)
        
        return averaged_weights
```
*FedProx adds a proximal term to the loss, helping stabilize training when client data is highly non-i.i.d.*

---

> **Try it yourself!** Simulate federated learning with different numbers of clients and data distributions. How does non-i.i.d. data affect convergence?

> **Key Insight:** Federated learning is a powerful paradigm for privacy-preserving, collaborative AI, but requires careful handling of data heterogeneity and communication efficiency.

### 3. Communication-Efficient Federated Learning

```python
class CommunicationEfficientTrainer:
    def __init__(self, global_model, num_clients=10, compression_ratio=0.1):
        self.global_model = global_model
        self.num_clients = num_clients
        self.compression_ratio = compression_ratio
        self.client_models = []
        
        # Initialize client models
        for _ in range(num_clients):
            client_model = type(global_model)()
            client_model.load_state_dict(global_model.state_dict())
            self.client_models.append(client_model)
    
    def compress_gradients(self, gradients):
        """Compress gradients using top-k sparsification."""
        compressed_gradients = []
        
        for grad in gradients:
            if grad is None:
                compressed_gradients.append(None)
                continue
            
            # Flatten gradient
            flat_grad = grad.view(-1)
            
            # Select top-k elements
            k = int(flat_grad.numel() * self.compression_ratio)
            _, indices = torch.topk(torch.abs(flat_grad), k)
            
            # Create sparse gradient
            compressed_grad = torch.zeros_like(flat_grad)
            compressed_grad[indices] = flat_grad[indices]
            
            compressed_gradients.append(compressed_grad.view_as(grad))
        
        return compressed_gradients
    
    def train_client_compressed(self, client_id, data_loader, epochs=1, lr=0.001):
        """Train client with gradient compression."""
        model = self.client_models[client_id]
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        accumulated_gradients = []
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Collect gradients
                gradients = [param.grad.clone() for param in model.parameters()]
                accumulated_gradients.append(gradients)
        
        # Average and compress gradients
        avg_gradients = []
        for param_idx in range(len(accumulated_gradients[0])):
            avg_grad = torch.zeros_like(accumulated_gradients[0][param_idx])
            for grad_list in accumulated_gradients:
                avg_grad += grad_list[param_idx]
            avg_grad /= len(accumulated_gradients)
            avg_gradients.append(avg_grad)
        
        # Compress gradients
        compressed_gradients = self.compress_gradients(avg_gradients)
        
        # Apply compressed gradients
        for param, comp_grad in zip(model.parameters(), compressed_gradients):
            if comp_grad is not None:
                param.data -= lr * comp_grad
        
        return model.state_dict()
```

## Privacy-Preserving Techniques

### 1. Differential Privacy

```python
import torch
import numpy as np

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0, delta=1e-5, sensitivity=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
    
    def add_noise(self, gradients):
        """Add noise to gradients for differential privacy."""
        # Calculate noise scale
        noise_scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        # Add Gaussian noise
        noisy_gradients = []
        for grad in gradients:
            if grad is not None:
                noise = torch.randn_like(grad) * noise_scale
                noisy_grad = grad + noise
                noisy_gradients.append(noisy_grad)
            else:
                noisy_gradients.append(None)
        
        return noisy_gradients
    
    def clip_gradients(self, gradients, clip_norm=1.0):
        """Clip gradients to bound sensitivity."""
        total_norm = 0
        for grad in gradients:
            if grad is not None:
                param_norm = grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        clip_coef = clip_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in gradients:
                if grad is not None:
                    grad.data.mul_(clip_coef)
        
        return gradients

class DPFederatedTrainer:
    def __init__(self, global_model, num_clients=10, epsilon=1.0, delta=1e-5):
        self.global_model = global_model
        self.num_clients = num_clients
        self.dp = DifferentialPrivacy(epsilon, delta)
        self.client_models = []
        
        # Initialize client models
        for _ in range(num_clients):
            client_model = type(global_model)()
            client_model.load_state_dict(global_model.state_dict())
            self.client_models.append(client_model)
    
    def train_client_dp(self, client_id, data_loader, epochs=1, lr=0.001):
        """Train client with differential privacy."""
        model = self.client_models[client_id]
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Clip gradients
                gradients = [param.grad for param in model.parameters()]
                clipped_gradients = self.dp.clip_gradients(gradients)
                
                # Add noise
                noisy_gradients = self.dp.add_noise(clipped_gradients)
                
                # Apply noisy gradients
                for param, noisy_grad in zip(model.parameters(), noisy_gradients):
                    if noisy_grad is not None:
                        param.grad = noisy_grad
                
                optimizer.step()
        
        return model.state_dict()
```

### 2. Secure Aggregation

```python
import hashlib
import secrets

class SecureAggregation:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.shared_keys = {}
    
    def generate_shared_keys(self):
        """Generate pairwise shared keys between clients."""
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                key = secrets.token_bytes(32)
                self.shared_keys[(i, j)] = key
                self.shared_keys[(j, i)] = key
    
    def mask_gradients(self, client_id, gradients):
        """Mask gradients using shared keys."""
        masked_gradients = []
        
        for grad in gradients:
            if grad is None:
                masked_gradients.append(None)
                continue
            
            # Create mask using shared keys
            mask = torch.zeros_like(grad)
            for other_client in range(self.num_clients):
                if other_client != client_id:
                    key = self.shared_keys[(client_id, other_client)]
                    # Generate pseudo-random mask from key
                    mask_seed = int.from_bytes(key, byteorder='big')
                    torch.manual_seed(mask_seed)
                    client_mask = torch.randn_like(grad)
                    
                    if client_id < other_client:
                        mask += client_mask
                    else:
                        mask -= client_mask
            
            masked_grad = grad + mask
            masked_gradients.append(masked_grad)
        
        return masked_gradients
    
    def unmask_gradients(self, masked_gradients_list):
        """Unmask aggregated gradients."""
        if not masked_gradients_list:
            return []
        
        num_params = len(masked_gradients_list[0])
        unmasked_gradients = [torch.zeros_like(masked_gradients_list[0][i]) 
                            for i in range(num_params)]
        
        # Sum all masked gradients
        for masked_gradients in masked_gradients_list:
            for i, masked_grad in enumerate(masked_gradients):
                if masked_grad is not None:
                    unmasked_gradients[i] += masked_grad
        
        # Masks cancel out in aggregation
        return unmasked_gradients
```

## Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Generate synthetic federated data
def generate_federated_data(num_clients=5, samples_per_client=1000):
    """Generate synthetic data for federated learning simulation."""
    client_data_loaders = []
    client_sizes = []
    
    for client_id in range(num_clients):
        # Generate data with some client-specific patterns
        np.random.seed(client_id)
        X = np.random.randn(samples_per_client, 784)
        y = np.random.randint(0, 10, samples_per_client)
        
        # Add some client-specific bias
        X += client_id * 0.1
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        client_data_loaders.append(dataloader)
        client_sizes.append(samples_per_client)
    
    return client_data_loaders, client_sizes

# Model definition
class FederatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Federated learning training
def train_federated_learning():
    # Generate data
    num_clients = 5
    client_data_loaders, client_sizes = generate_federated_data(num_clients)
    
    # Initialize global model
    global_model = FederatedModel()
    
    # Initialize federated trainer
    trainer = FederatedTrainer(global_model, num_clients)
    
    # Training rounds
    num_rounds = 20
    for round_id in range(num_rounds):
        print(f"Training round {round_id + 1}/{num_rounds}")
        
        # Train one round
        trainer.train_round(client_data_loaders, client_sizes)
        
        # Evaluate global model (simplified)
        if round_id % 5 == 0:
            print(f"Round {round_id + 1} completed")
    
    return global_model

# Run federated learning
if __name__ == "__main__":
    trained_model = train_federated_learning()
    print("Federated learning completed!")
```

## Best Practices

### 1. Client Selection
- Implement client sampling strategies
- Handle stragglers and failures
- Balance client participation

### 2. Communication Optimization
- Use gradient compression
- Implement asynchronous updates
- Optimize aggregation frequency

### 3. Privacy Protection
- Apply differential privacy
- Use secure aggregation
- Implement data anonymization

### 4. Robustness
- Handle Byzantine attacks
- Implement fault tolerance
- Monitor system health

## Summary

Federated learning enables collaborative model training while preserving data privacy. Key considerations:

1. **Privacy preservation** through local training and secure aggregation
2. **Communication efficiency** to handle network constraints
3. **Robustness** against failures and attacks
4. **Scalability** to large numbers of clients
5. **Fairness** in client participation and contribution

Proper implementation can achieve high-quality models while maintaining data privacy and regulatory compliance. 