# Sequential Data Processing

## Overview

Sequential data processing is the foundation of Recurrent Neural Networks (RNNs). Unlike traditional neural networks that process fixed-size inputs independently, RNNs are designed to handle variable-length sequences while maintaining context across time steps.

## What is Sequential Data?

Sequential data consists of ordered collections where the position and order of elements matter. Examples include:

- **Text**: Words in a sentence, characters in a word
- **Time series**: Stock prices, weather data, sensor readings
- **Audio**: Speech signals, music
- **Video**: Frames in a video sequence
- **DNA sequences**: Genetic information

## Key Characteristics

### 1. Variable Length
Sequences can have different lengths, making traditional fixed-input neural networks unsuitable.

### 2. Temporal Dependencies
Information at time step $`t`$ often depends on previous time steps $`t-1, t-2, \ldots`$.

### 3. Order Matters
The sequence $`[A, B, C]`$ is different from $`[C, B, A]`$.

## Mathematical Representation

A sequence of length $`T`$ can be represented as:

```math
X = (x_1, x_2, \ldots, x_T)
```

Where each $`x_t`$ is a vector of features at time step $`t`$.

## Memory Mechanisms

### Hidden State
The core concept in RNNs is the hidden state $`h_t`$, which serves as the network's memory:

```math
h_t = f(h_{t-1}, x_t)
```

Where $`f`$ is a function that combines the previous hidden state with the current input.

### Information Flow
The hidden state carries information from the beginning of the sequence to the current time step:

```math
h_t = f(f(f(\ldots f(h_0, x_1), x_2), \ldots), x_t)
```

## Python Implementation

### Basic Sequence Processing

```python
import numpy as np
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Weight matrices
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_hy = nn.Linear(hidden_size, output_size)
        
        # Activation function
        self.tanh = nn.Tanh()
        
    def forward(self, x, h0=None):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()
        
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size)
        
        # Initialize hidden states
        h = h0
        outputs = []
        
        # Process each time step
        for t in range(seq_len):
            # Current input
            x_t = x[:, t, :]
            
            # Update hidden state
            h = self.tanh(self.W_xh(x_t) + self.W_hh(h))
            
            # Generate output
            y_t = self.W_hy(h)
            outputs.append(y_t)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        return outputs, h

# Example usage
input_size = 10
hidden_size = 20
output_size = 5
batch_size = 3
seq_len = 7

# Create model
rnn = SimpleRNN(input_size, hidden_size, output_size)

# Create sample data
x = torch.randn(batch_size, seq_len, input_size)

# Forward pass
outputs, final_hidden = rnn(x)
print(f"Output shape: {outputs.shape}")
print(f"Final hidden state shape: {final_hidden.shape}")
```

### Variable Length Sequence Handling

```python
import torch.nn.utils.rnn as rnn_utils

class VariableLengthRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VariableLengthRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, sequences, lengths):
        # Pack sequences
        packed = rnn_utils.pack_padded_sequence(
            sequences, lengths, batch_first=True, enforce_sorted=False
        )
        
        # RNN forward pass
        packed_output, hidden = self.rnn(packed)
        
        # Unpack sequences
        output, _ = rnn_utils.pad_packed_sequence(packed_output, batch_first=True)
        
        # Apply output layer
        output = self.output_layer(output)
        
        return output, hidden

# Example with variable length sequences
def create_variable_length_data():
    # Create sequences of different lengths
    sequences = [
        torch.randn(4, 10),  # Length 4
        torch.randn(6, 10),  # Length 6
        torch.randn(3, 10),  # Length 3
    ]
    
    # Get lengths
    lengths = [seq.size(0) for seq in sequences]
    
    # Pad sequences
    padded = rnn_utils.pad_sequence(sequences, batch_first=True)
    
    return padded, lengths

# Usage
padded_sequences, lengths = create_variable_length_data()
model = VariableLengthRNN(10, 20, 5)
output, hidden = model(padded_sequences, lengths)
```

## Sequence Modeling Tasks

### 1. Sequence Classification
Classify entire sequences into categories.

```python
class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SequenceClassifier, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, lengths):
        # Pack and process
        packed = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        _, (hidden, _) = self.rnn(packed)
        
        # Use final hidden state for classification
        final_hidden = hidden[-1]  # Last layer's hidden state
        output = self.classifier(final_hidden)
        return output
```

### 2. Sequence Generation
Generate sequences one element at a time.

```python
class SequenceGenerator(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_size):
        super(SequenceGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h0=None):
        # Embed input
        embedded = self.embedding(x)
        
        # RNN forward pass
        output, (h, c) = self.rnn(embedded, h0)
        
        # Generate predictions
        predictions = self.output_layer(output)
        return predictions, (h, c)
    
    def generate(self, start_token, max_length, temperature=1.0):
        self.eval()
        with torch.no_grad():
            # Initialize
            current_token = torch.tensor([[start_token]])
            h = None
            generated = [start_token]
            
            for _ in range(max_length):
                # Forward pass
                output, (h, _) = self(current_token, h)
                
                # Sample next token
                logits = output[0, -1] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                generated.append(next_token.item())
                current_token = next_token.unsqueeze(0)
                
        return generated
```

## Data Preprocessing

### Text Sequence Processing

```python
import re
from collections import Counter

class TextProcessor:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.vocab_size = 4
        
    def build_vocab(self, texts):
        # Count words
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Build vocabulary
        for word, count in word_counts.items():
            if count >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
    
    def text_to_sequence(self, text, max_length=None):
        words = text.lower().split()
        sequence = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        if max_length:
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            else:
                sequence += [self.word2idx['<PAD>']] * (max_length - len(sequence))
        
        return sequence
    
    def sequence_to_text(self, sequence):
        words = [self.idx2word[idx] for idx in sequence if idx not in [0, 1, 2, 3]]
        return ' '.join(words)

# Example usage
texts = [
    "hello world this is a test",
    "another example sentence",
    "more text for vocabulary building"
]

processor = TextProcessor(min_freq=1)
processor.build_vocab(texts)

# Convert text to sequence
sequence = processor.text_to_sequence("hello world", max_length=10)
print(f"Sequence: {sequence}")

# Convert back to text
text = processor.sequence_to_text(sequence)
print(f"Text: {text}")
```

### Time Series Processing

```python
class TimeSeriesProcessor:
    def __init__(self, window_size=10, stride=1):
        self.window_size = window_size
        self.stride = stride
        
    def create_sequences(self, data):
        """Create sliding window sequences from time series data."""
        sequences = []
        targets = []
        
        for i in range(0, len(data) - self.window_size, self.stride):
            sequence = data[i:i + self.window_size]
            target = data[i + self.window_size]
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def normalize(self, data, method='minmax'):
        """Normalize time series data."""
        if method == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = np.mean(data)
            std_val = np.std(data)
            return (data - mean_val) / std_val
        else:
            raise ValueError(f"Unknown normalization method: {method}")

# Example with synthetic time series
def generate_synthetic_timeseries(n_points=1000):
    """Generate synthetic time series data."""
    t = np.linspace(0, 10, n_points)
    # Combine multiple frequencies
    signal = (np.sin(2 * np.pi * t) + 
              0.5 * np.sin(2 * np.pi * 3 * t) + 
              0.3 * np.random.randn(n_points))
    return signal

# Usage
data = generate_synthetic_timeseries()
processor = TimeSeriesProcessor(window_size=20, stride=1)
sequences, targets = processor.create_sequences(data)

print(f"Original data shape: {data.shape}")
print(f"Sequences shape: {sequences.shape}")
print(f"Targets shape: {targets.shape}")
```

## Performance Considerations

### 1. Memory Efficiency
- Use packed sequences for variable length data
- Implement gradient checkpointing for long sequences
- Consider using smaller hidden states

### 2. Computational Efficiency
- Batch processing for parallel computation
- Use appropriate data types (float16 for inference)
- Optimize sequence length based on task requirements

### 3. Training Stability
- Gradient clipping to prevent exploding gradients
- Proper initialization of RNN weights
- Learning rate scheduling

## Summary

Sequential data processing is fundamental to RNNs and involves:

1. **Understanding sequence characteristics**: variable length, temporal dependencies, order importance
2. **Memory mechanisms**: hidden states that carry information across time steps
3. **Proper data preprocessing**: handling variable lengths, normalization, vocabulary building
4. **Task-specific architectures**: classification, generation, prediction
5. **Performance optimization**: memory and computational efficiency

The key insight is that RNNs maintain an internal state that evolves as the sequence is processed, allowing them to capture temporal dependencies and handle variable-length inputs effectively. 