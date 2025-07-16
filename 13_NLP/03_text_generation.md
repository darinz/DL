# Text Generation

Text generation is the task of producing coherent and contextually relevant text given an input prompt. It is a core capability of modern language models and is used in applications such as machine translation, summarization, and creative writing.

---

## 1. Autoregressive Text Generation

Autoregressive models generate text one token at a time, conditioning each token on the previous ones.

$`P(x) = \prod_{t=1}^T P(x_t \mid x_{<t})`$

Where $x$ is a sequence of tokens.

**Explanation:**
- The model predicts the probability of each token $x_t$ given all previous tokens $x_{<t}$.
- The product over $t$ means the probability of the whole sequence is the product of the probabilities of each token, conditioned on the previous ones.
- This is the basis for models like GPT-2 and GPT-3.

---

## 2. Sequence-to-Sequence Models

Sequence-to-sequence (seq2seq) models encode an input sequence $x$ to a context vector, then decode it to an output sequence $y$.

```math
P(y \mid x) = \prod_{t=1}^{T'} P(y_t \mid y_{<t}, x)
```

**Explanation:**
- The encoder processes the input $x$ and produces a context (summary) vector.
- The decoder generates each output token $y_t$ based on the context and previously generated tokens $y_{<t}$.
- Used in tasks like translation (input: English, output: French).

---

## 3. Example: Text Generation with GPT-2

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")  # Load pre-trained GPT-2 model

# Load tokenizer to convert text to tokens
# (tokenizer and model must match)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")  # Encode prompt as tensor
output = model.generate(input_ids, max_length=50, num_return_sequences=1)  # Generate text
print(tokenizer.decode(output[0], skip_special_tokens=True))  # Decode tokens to text
```

**Code Annotations:**
- `GPT2LMHeadModel` loads the GPT-2 language model for text generation.
- `GPT2Tokenizer` converts text to token IDs and back.
- The prompt is encoded into input IDs suitable for the model.
- `model.generate` produces a sequence of tokens extending the prompt.
- The output is decoded back into human-readable text.

> **Tip:** You can experiment with `max_length` and decoding strategies (see below) to control the output.

---

## 4. Decoding Strategies
- **Greedy decoding:** Selects the most probable token at each step. Fast but can be repetitive or bland.
- **Beam search:** Keeps multiple hypotheses at each step, balancing exploration and exploitation. Produces more coherent text but is slower.
- **Sampling:** Samples from the probability distribution over tokens, introducing randomness and diversity.
- **Top-k and Top-p (nucleus) sampling:**
    - **Top-k:** Restricts sampling to the top $k$ most probable tokens.
    - **Top-p (nucleus):** Samples from the smallest set of tokens whose cumulative probability exceeds $p$ (e.g., $p=0.9$).
    - These methods help balance creativity and coherence.

---

## 5. Applications
- **Machine translation:** Translating text from one language to another.
- **Summarization:** Condensing long documents into shorter summaries.
- **Dialogue systems:** Powering chatbots and conversational agents.
- **Story and code generation:** Creating stories, articles, or code from prompts.

---

## 6. Further Reading
- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [A Neural Conversational Model (Vinyals & Le, 2015)](https://arxiv.org/abs/1506.05869) 

> **Explore these papers to learn more about the foundations and advances in text generation!** 