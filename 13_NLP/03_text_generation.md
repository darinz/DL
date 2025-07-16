# Text Generation

Text generation is the task of producing coherent and contextually relevant text given an input prompt. It is a core capability of modern language models and is used in applications such as machine translation, summarization, and creative writing.

## 1. Autoregressive Text Generation

Autoregressive models generate text one token at a time, conditioning each token on the previous ones.

$`P(x) = \prod_{t=1}^T P(x_t \mid x_{<t})`$

Where $`x`$ is a sequence of tokens.

## 2. Sequence-to-Sequence Models

Sequence-to-sequence (seq2seq) models encode an input sequence $`x`$ to a context vector, then decode it to an output sequence $`y`$.

```math
P(y \mid x) = \prod_{t=1}^{T'} P(y_t \mid y_{<t}, x)
```

## 3. Example: Text Generation with GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 4. Decoding Strategies
- **Greedy decoding:** Selects the most probable token at each step.
- **Beam search:** Keeps multiple hypotheses at each step.
- **Sampling:** Samples from the probability distribution over tokens.
- **Top-k and Top-p (nucleus) sampling:** Restricts sampling to the top $`k`$ tokens or the smallest set whose cumulative probability exceeds $`p`$.

## 5. Applications
- Machine translation
- Summarization
- Dialogue systems
- Story and code generation

## 6. Further Reading
- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [A Neural Conversational Model (Vinyals & Le, 2015)](https://arxiv.org/abs/1506.05869) 