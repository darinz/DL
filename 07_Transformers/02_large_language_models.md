# Large Language Models: BERT, GPT, T5

This guide covers three influential large language models based on the Transformer architecture: BERT, GPT, and T5. Each model has unique design choices and training objectives, making them suitable for different NLP tasks.

## 1. BERT (Bidirectional Encoder Representations from Transformers)

BERT uses only the encoder part of the Transformer and is trained to understand bidirectional context.

### 1.1. Masked Language Modeling (MLM)
BERT randomly masks some tokens in the input and trains the model to predict them:

```math
\text{Loss}_{MLM} = -\sum_{i \in M} \log P(x_i | X_{\setminus M})
```
where $`M`$ is the set of masked positions.

### 1.2. Next Sentence Prediction (NSP)
BERT is also trained to predict if one sentence follows another:

```math
\text{Loss}_{NSP} = -\log P(y | A, B)
```
where $`y`$ is a binary label for whether sentence $`B`$ follows $`A`$.

### 1.3. Python Example: Using HuggingFace Transformers
```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # (1, seq_len, hidden_size)
```

## 2. GPT (Generative Pre-trained Transformer)

GPT uses only the decoder part of the Transformer and is trained with left-to-right language modeling.

### 2.1. Language Modeling Objective

```math
\text{Loss}_{LM} = -\sum_{t=1}^T \log P(x_t | x_{<t})
```

### 2.2. Python Example: Text Generation with GPT-2
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=20)
print(tokenizer.decode(outputs[0]))
```

## 3. T5 (Text-to-Text Transfer Transformer)

T5 frames every NLP task as a text-to-text problem, using an encoder-decoder architecture.

### 3.1. Unified Text-to-Text Framework
T5 is trained on a mixture of tasks, all cast as text-to-text:

- Translation: "translate English to German: ..."
- Summarization: "summarize: ..."
- Classification: "cola sentence: ..."

### 3.2. Python Example: Using T5 for Summarization
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
inputs = tokenizer("summarize: The quick brown fox jumps over the lazy dog.", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## 4. Summary

- **BERT:** Bidirectional, encoder-only, masked language modeling.
- **GPT:** Unidirectional, decoder-only, left-to-right generation.
- **T5:** Encoder-decoder, text-to-text framework for all tasks.

For more, see the original papers:
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [T5](https://arxiv.org/abs/1910.10683) 