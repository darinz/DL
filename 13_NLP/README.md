# Natural Language Processing (NLP)

[![NLP](https://img.shields.io/badge/NLP-Natural%20Language-blue?style=for-the-badge&logo=language)](https://github.com/yourusername/DL)
[![LLMs](https://img.shields.io/badge/LLMs-Large%20Language-green?style=for-the-badge&logo=robot)](https://github.com/yourusername/DL/tree/main/13_NLP)
[![Multimodal](https://img.shields.io/badge/Multimodal-Text%20Image-orange?style=for-the-badge&logo=link)](https://github.com/yourusername/DL/tree/main/13_NLP)
[![Text Generation](https://img.shields.io/badge/Text%20Generation-GPT%20BART-purple?style=for-the-badge&logo=pen-fancy)](https://github.com/yourusername/DL/tree/main/13_NLP)
[![Question Answering](https://img.shields.io/badge/Question%20Answering-BERT%20RoBERTa-red?style=for-the-badge&logo=question-circle)](https://github.com/yourusername/DL/tree/main/13_NLP)
[![CLIP](https://img.shields.io/badge/CLIP-Contrastive%20Learning-yellow?style=for-the-badge&logo=eye)](https://github.com/yourusername/DL/tree/main/13_NLP)
[![DALL-E](https://img.shields.io/badge/DALL-E-Text%20to%20Image-blue?style=for-the-badge&logo=magic)](https://github.com/yourusername/DL/tree/main/13_NLP)
[![Transformers](https://img.shields.io/badge/Transformers-Self%20Attention-orange?style=for-the-badge&logo=bolt)](https://github.com/yourusername/DL/tree/main/13_NLP)

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

```math
P(x) = \prod_{t=1}^T P(x_t \mid x_{\lt t})
```

- **Sequence-to-Sequence Models:**
  - Encode input sequence $`x`$ to a context vector, then decode to output sequence $`y`$.

### 4. [Question Answering](04_question_answering.md)
- **Examples:** BERT, RoBERTa
- Question Answering (QA) systems extract or generate answers to questions posed in natural language. Modern QA models use pre-trained language models fine-tuned on QA datasets.
- **Extractive QA:**
  - Given a context $`C`$ and question $`Q`$, predict the span $`(s, e)`$ in $`C`$ that answers $`Q`$.
- **Mathematical Formulation:**

```math
(s^ \ast , e^ \ast) = \underset{(s, e)}{\arg\max}\; P(s, e \mid C, Q)
```