# Natural Language Processing (NLP)

Natural Language Processing (NLP) is a field at the intersection of computer science, artificial intelligence, and linguistics, focused on enabling machines to understand, interpret, and generate human language. This section covers foundational and advanced topics in NLP, with a focus on modern deep learning approaches.

## Topics Covered

### 1. [Large Language Models](01_large_language_models.md)
- **Examples:** GPT-4, Claude, LLaMA
- Large Language Models (LLMs) are deep neural networks trained on vast corpora of text data. They are capable of understanding and generating human-like text, performing tasks such as translation, summarization, and dialogue.
- LLMs are typically based on the Transformer architecture, which uses self-attention mechanisms to model long-range dependencies in text.
- **Mathematical Formulation:**

```math
\text{Self-Attention:}\quad \mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### 2. [Multimodal Models](02_multimodal_models.md)
- **Examples:** CLIP, DALL-E, Flamingo
- Multimodal models process and relate information from multiple modalities, such as text and images. They learn joint representations that enable tasks like image captioning, visual question answering, and text-to-image generation.
- These models often use contrastive learning or generative objectives to align modalities in a shared embedding space.

### 3. [Text Generation](03_text_generation.md)
- **Examples:** T5, BART, GPT
- Text generation models produce coherent and contextually relevant text given an input prompt. They are used for tasks such as machine translation, summarization, and creative writing.
- **Autoregressive Generation:**

$`P(x) = \prod_{t=1}^T P(x_t \mid x_{<t})`$

- **Sequence-to-Sequence Models:**
  - Encode input sequence $`x`$ to a context vector, then decode to output sequence $`y`$.

### 4. [Question Answering](04_question_answering.md)
- **Examples:** BERT, RoBERTa
- Question Answering (QA) systems extract or generate answers to questions posed in natural language. Modern QA models use pre-trained language models fine-tuned on QA datasets.
- **Extractive QA:**
  - Given a context $`C`$ and question $`Q`$, predict the span $`(s, e)`$ in $`C`$ that answers $`Q`$.
- **Mathematical Formulation:**

$`(s^*, e^*) = \underset{(s, e)}{\arg\max}\; P(s, e \mid C, Q)`$

---

This directory contains resources, notes, and code related to these NLP topics, with mathematical concepts presented using LaTeX for clarity. 