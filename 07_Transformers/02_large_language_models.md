# Large Language Models: BERT, GPT, T5

> **Key Insight:** BERT, GPT, and T5 are foundational models that demonstrate the flexibility of the Transformer architecture for a wide range of NLP tasks.

This guide covers three influential large language models based on the Transformer architecture: BERT, GPT, and T5. Each model has unique design choices and training objectives, making them suitable for different NLP tasks.

## 1. BERT (Bidirectional Encoder Representations from Transformers)

BERT uses only the encoder part of the Transformer and is trained to understand bidirectional context.

> **Did you know?** BERT's bidirectional training allows it to capture context from both the left and right of each token, unlike traditional left-to-right models.

### 1.1. Masked Language Modeling (MLM)
BERT randomly masks some tokens in the input and trains the model to predict them:

```math
\text{Loss}_{MLM} = -\sum_{i \in M} \log P(x_i | X_{\setminus M})
```
where $`M`$ is the set of masked positions.

#### Intuitive Explanation

By masking out words and asking the model to fill them in, BERT learns deep bidirectional representations of language.

### 1.2. Next Sentence Prediction (NSP)
BERT is also trained to predict if one sentence follows another:

```math
\text{Loss}_{NSP} = -\log P(y | A, B)
```
where $`y`$ is a binary label for whether sentence $`B`$ follows $`A`$.

> **Common Pitfall:** When fine-tuning BERT, always preprocess your data to match the pretraining objectives (e.g., use [CLS] and [SEP] tokens).

### 1.3. Python Example: Using HuggingFace Transformers
```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # (1, seq_len, hidden_size)
```

> **Try it yourself!** Mask different words in a sentence and see how BERT predicts them.

## 2. GPT (Generative Pre-trained Transformer)

GPT uses only the decoder part of the Transformer and is trained with left-to-right language modeling.

> **Key Insight:** GPT's autoregressive training makes it ideal for text generation, as it predicts the next word given previous words.

### 2.1. Language Modeling Objective

```math
\text{Loss}_{LM} = -\sum_{t=1}^T \log P(x_t | x_{<t})
```

#### Intuitive Explanation

GPT learns to generate coherent text by predicting the next word in a sequence, one word at a time.

### 2.2. Python Example: Text Generation with GPT-2
```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
inputs = tokenizer("Once upon a time", return_tensors="pt")
outputs = model.generate(**inputs, max_length=20)
print(tokenizer.decode(outputs[0]))
```

> **Did you know?** GPT-2 and GPT-3 can generate entire articles, stories, and even code by sampling one token at a time.

## 3. T5 (Text-to-Text Transfer Transformer)

T5 frames every NLP task as a text-to-text problem, using an encoder-decoder architecture.

> **Key Insight:** By casting all tasks as text-to-text, T5 can be fine-tuned for translation, summarization, classification, and more with a single model.

### 3.1. Unified Text-to-Text Framework
T5 is trained on a mixture of tasks, all cast as text-to-text:

- Translation: "translate English to German: ..."
- Summarization: "summarize: ..."
- Classification: "cola sentence: ..."

#### Intuitive Explanation

T5's flexible framework allows it to handle any NLP task where both input and output can be represented as text.

### 3.2. Python Example: Using T5 for Summarization
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
inputs = tokenizer("summarize: The quick brown fox jumps over the lazy dog.", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

> **Try it yourself!** Use T5 to translate, summarize, or classify text by changing the input prompt.

## 4. Summary & Next Steps

| Model | Architecture | Training Objective | Typical Use Cases |
|-------|--------------|-------------------|------------------|
| BERT  | Encoder-only | MLM + NSP         | Classification, QA, NER |
| GPT   | Decoder-only | Left-to-right LM  | Text generation, completion |
| T5    | Encoder-Decoder | Text-to-text    | Translation, summarization, multi-task |

- **BERT:** Bidirectional, encoder-only, masked language modeling.
- **GPT:** Unidirectional, decoder-only, left-to-right generation.
- **T5:** Encoder-decoder, text-to-text framework for all tasks.

> **Key Insight:** Understanding the differences between BERT, GPT, and T5 is crucial for selecting the right model for your NLP application.

### Next Steps
- Fine-tune BERT, GPT, or T5 on your own dataset for a downstream task.
- Explore the HuggingFace Transformers library for more models and tasks.
- Read the original papers:
  - [BERT](https://arxiv.org/abs/1810.04805)
  - [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
  - [T5](https://arxiv.org/abs/1910.10683)

> **Did you know?** The largest T5 and GPT models have hundreds of billions of parameters, enabling them to perform few-shot and zero-shot learning. 