# Text Generation

Text generation is the task of producing coherent and contextually relevant text given an input prompt. It is a core capability of modern language models and is used in applications such as machine translation, summarization, and creative writing.

> **Learning Objective:** By the end of this guide, you'll understand different text generation approaches, decoding strategies, and how to implement text generation systems using modern language models.

---

## 1. Autoregressive Text Generation

Autoregressive models generate text one token at a time, conditioning each token on the previous ones.

```math
P(x) = \prod_{t=1}^T P(x_t \mid x_{<t})
```

Where $x$ is a sequence of tokens.

**Mathematical Intuition:**

1. **Conditional Probability:** Each token depends on all previous tokens
   ```math
   P(x_t \mid x_{<t}) = P(x_t \mid x_1, x_2, ..., x_{t-1})
   ```

2. **Chain Rule:** Joint probability is product of conditionals
   ```math
   P(x_1, x_2, ..., x_T) = P(x_1) \times P(x_2 \mid x_1) \times ... \times P(x_T \mid x_1, ..., x_{T-1})
   ```

3. **Autoregressive Property:** Model "reads" from left to right

**Example with "The cat sat":**
```math
P(\text{"The cat sat"}) = P(\text{"The"}) \times P(\text{"cat"} \mid \text{"The"}) \times P(\text{"sat"} \mid \text{"The cat"})
```

**Why Autoregressive Generation Works:**
- **Natural Language Structure:** Words depend on context
- **Unsupervised Learning:** No manual labeling required
- **Scalable:** Can use massive amounts of text data
- **Flexible:** Can generate text of varying lengths

**Challenges:**
- **Sequential Nature:** Can't parallelize generation
- **Error Propagation:** Early mistakes affect later tokens
- **Repetition:** Models may get stuck in loops
- **Coherence:** Long-range dependencies are difficult

**Implementation Concept:**
```python
def autoregressive_generate(model, prompt, max_length=50):
    """
    Generate text autoregressively
    
    Args:
        model: Language model
        prompt: Input text prompt
        max_length: Maximum length to generate
    """
    tokens = tokenize(prompt)
    
    for _ in range(max_length):
        # Get model predictions for next token
        logits = model(tokens)
        next_token_probs = softmax(logits[-1])
        
        # Sample next token (could use different strategies)
        next_token = sample_from_distribution(next_token_probs)
        
        # Add to sequence
        tokens.append(next_token)
        
        # Stop if end-of-sequence token
        if next_token == EOS_TOKEN:
            break
    
    return detokenize(tokens)
```

---

## 2. Sequence-to-Sequence Models

Sequence-to-sequence (seq2seq) models encode an input sequence $x$ to a context vector, then decode it to an output sequence $y$.

```math
P(y \mid x) = \prod_{t=1}^{T'} P(y_t \mid y_{\lt t}, x)
```

**Architecture Breakdown:**

1. **Encoder:** Processes input sequence into context vector
   ```math
   h = \text{Encoder}(x_1, x_2, ..., x_T)
   ```

2. **Decoder:** Generates output conditioned on context
   ```math
   P(y_t \mid y_{<t}, x) = \text{Decoder}(y_{<t}, h)
   ```

3. **Attention Mechanism:** Helps decoder focus on relevant parts of input
   ```math
   \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   ```

**Use Cases:**
- **Machine Translation:** English → French
- **Summarization:** Long text → Short summary
- **Question Answering:** Question + Context → Answer
- **Dialogue Systems:** User input → Bot response

**Advantages over Autoregressive:**
- **Conditioned Generation:** Output depends on specific input
- **Bidirectional Context:** Encoder can see full input
- **Task-Specific:** Can be optimized for specific tasks

**Limitations:**
- **Fixed Context:** Limited by encoder capacity
- **Training Complexity:** More complex than language models
- **Inference Speed:** Slower than autoregressive models

**Implementation Example:**
```python
class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.encoder = nn.LSTM(vocab_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(vocab_size, hidden_size, batch_first=True)
        self.output_projection = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_seq, target_seq=None, max_length=50):
        # Encode input
        encoder_output, (hidden, cell) = self.encoder(input_seq)
        
        # Initialize decoder
        decoder_input = torch.zeros(input_seq.size(0), 1, vocab_size)
        decoder_hidden = (hidden, cell)
        
        outputs = []
        for t in range(max_length):
            # Decode one step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Project to vocabulary
            output = self.output_projection(decoder_output)
            outputs.append(output)
            
            # Next input (teacher forcing during training)
            if target_seq is not None and t < target_seq.size(1):
                decoder_input = target_seq[:, t:t+1]
            else:
                decoder_input = output.argmax(dim=-1, keepdim=True)
        
        return torch.cat(outputs, dim=1)
```

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

**Detailed Code Walkthrough:**

```python
# Step 1: Import and Setup
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Step 2: Load Model and Tokenizer
model_name = "gpt2"  # Options: "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 3: Prepare Input
prompt = "Once upon a time"
# Tokenization converts text to numbers the model understands
# Example: "Once upon a time" → [5112, 338, 318, 257]
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Step 4: Generate Text
# Basic generation with default parameters
output = model.generate(
    input_ids,
    max_length=50,           # Maximum number of tokens to generate
    num_return_sequences=1,  # Number of different continuations
    do_sample=False,         # Use greedy decoding (deterministic)
    pad_token_id=tokenizer.eos_token_id  # Use EOS token for padding
)

# Step 5: Decode Output
# skip_special_tokens=True removes tokens like [PAD], [SEP], etc.
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")
```

**Try It Yourself:**
```python
# Experiment with different prompts
def generate_story(prompt, max_length=100):
    """Generate a story continuation from a prompt"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,      # Use sampling for more creative output
        temperature=0.8,     # Control randomness
        top_k=50,           # Only consider top 50 tokens
        repetition_penalty=1.2  # Penalize repetition
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test different story starters
story_prompts = [
    "In a distant galaxy",
    "The old wizard opened",
    "Deep in the forest",
    "The robot looked at"
]

for prompt in story_prompts:
    story = generate_story(prompt)
    print(f"Prompt: {prompt}")
    print(f"Story: {story}\n")
```

**Advanced Generation Parameters:**
```python
# More sophisticated generation with multiple strategies
def advanced_generation(prompt, strategy="sampling"):
    """Generate text using different strategies"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    if strategy == "greedy":
        # Always choose most likely token
        output = model.generate(
            input_ids,
            max_length=100,
            do_sample=False,
            num_return_sequences=1
        )
    elif strategy == "sampling":
        # Sample from probability distribution
        output = model.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            temperature=0.8,
            num_return_sequences=3  # Generate multiple options
        )
    elif strategy == "beam_search":
        # Use beam search for better coherence
        output = model.generate(
            input_ids,
            max_length=100,
            do_sample=False,
            num_beams=5,           # Number of beams
            early_stopping=True,   # Stop when all beams reach EOS
            num_return_sequences=3
        )
    elif strategy == "nucleus":
        # Nucleus sampling (top-p)
        output = model.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            top_p=0.9,            # Consider tokens until cumulative prob reaches 0.9
            temperature=0.7,
            num_return_sequences=3
        )
    
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

# Test different strategies
prompt = "The future of artificial intelligence"
strategies = ["greedy", "sampling", "beam_search", "nucleus"]

for strategy in strategies:
    print(f"\n=== {strategy.upper()} ===")
    results = advanced_generation(prompt, strategy)
    for i, result in enumerate(results):
        print(f"{i+1}. {result}")
```

> **Pro Tip:** Start with `temperature=0.7` and `do_sample=True` for creative text, or `temperature=0.0` for more predictable outputs.

---

## 4. Decoding Strategies

**Greedy Decoding:**
- **Strategy:** Always select the most probable token
- **Pros:** Fast, deterministic
- **Cons:** Can be repetitive, lacks creativity
- **Use Case:** When you need consistent, predictable output

```python
def greedy_decode(model, input_ids, max_length=50):
    """Greedy decoding implementation"""
    for _ in range(max_length):
        # Get model predictions
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Select most probable token
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Append to sequence
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Stop if EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return input_ids
```

**Beam Search:**
- **Strategy:** Keep multiple hypotheses at each step
- **Pros:** More coherent, better quality
- **Cons:** Slower, less diverse
- **Use Case:** When quality is more important than speed

```python
def beam_search_decode(model, input_ids, beam_width=5, max_length=50):
    """Beam search decoding implementation"""
    # Initialize beams
    beams = [(input_ids, 0.0)]  # (sequence, score)
    
    for _ in range(max_length):
        new_beams = []
        
        for beam_seq, beam_score in beams:
            # Get predictions for current beam
            outputs = model(beam_seq)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get top-k tokens
            top_k_probs, top_k_tokens = torch.topk(next_token_probs, beam_width)
            
            for i in range(beam_width):
                next_token = top_k_tokens[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([beam_seq, next_token], dim=-1)
                new_score = beam_score + torch.log(top_k_probs[0, i])
                new_beams.append((new_seq, new_score))
        
        # Keep top beam_width beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]
    
    # Return best beam
    return beams[0][0]
```

**Sampling Strategies:**

**Top-k Sampling:**
```python
def top_k_sampling(model, input_ids, k=50, max_length=50):
    """Top-k sampling implementation"""
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Get top-k tokens
        top_k_logits, top_k_indices = torch.topk(next_token_logits, k)
        
        # Sample from top-k
        probs = torch.softmax(top_k_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, 1)
        next_token = top_k_indices[0, sampled_idx].unsqueeze(0)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return input_ids
```

**Top-p (Nucleus) Sampling:**
```python
def nucleus_sampling(model, input_ids, p=0.9, max_length=50):
    """Nucleus sampling implementation"""
    for _ in range(max_length):
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        
        # Sort probabilities
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        
        # Find cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find tokens to keep (cumulative prob <= p)
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Remove low-probability tokens
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample from remaining tokens
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return input_ids
```

**Strategy Comparison:**
| Strategy | Speed | Quality | Diversity | Use Case |
|----------|-------|---------|-----------|----------|
| Greedy | Fast | Medium | Low | Factual text |
| Beam Search | Slow | High | Low | Translation |
| Sampling | Medium | Medium | High | Creative writing |
| Top-k | Medium | Medium | Medium | Balanced |
| Top-p | Medium | Medium | High | Creative tasks |

---

## 5. Applications

**Machine Translation:**
- **Language Pairs:** English ↔ French, Spanish, German, etc.
- **Neural MT:** End-to-end translation using seq2seq models
- **Quality Metrics:** BLEU, METEOR, ROUGE scores

**Summarization:**
- **Extractive:** Select important sentences from source
- **Abstractive:** Generate new summary text
- **Applications:** News summarization, document compression

**Dialogue Systems:**
- **Chatbots:** Customer service, entertainment
- **Virtual Assistants:** Task completion, information retrieval
- **Conversational AI:** Natural language interaction

**Story and Code Generation:**
- **Creative Writing:** Stories, poems, articles
- **Code Generation:** Programming assistance, autocomplete
- **Content Creation:** Marketing copy, product descriptions

**Example Application: Creative Writing Assistant**
```python
def creative_writing_assistant(prompt, style="story", max_length=200):
    """Generate creative text in different styles"""
    
    # Style-specific prompts
    style_prompts = {
        "story": f"Write a creative story: {prompt}",
        "poem": f"Write a poem about: {prompt}",
        "article": f"Write an informative article about: {prompt}",
        "dialogue": f"Write a dialogue about: {prompt}"
    }
    
    input_text = style_prompts.get(style, prompt)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Use creative generation parameters
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.9,        # High temperature for creativity
        top_p=0.9,             # Nucleus sampling
        repetition_penalty=1.3, # Avoid repetition
        num_return_sequences=3  # Generate multiple options
    )
    
    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]

# Test the creative writing assistant
prompts = [
    "a robot learning to paint",
    "a magical library",
    "time travel paradox"
]

for prompt in prompts:
    print(f"\n=== Story about: {prompt} ===")
    stories = creative_writing_assistant(prompt, style="story")
    for i, story in enumerate(stories):
        print(f"\nOption {i+1}:")
        print(story)
```

**Example Application: Code Generation**
```python
def code_generator(description, language="python", max_length=150):
    """Generate code from natural language description"""
    
    # Language-specific prompts
    lang_prompts = {
        "python": f"# Python function: {description}\ndef",
        "javascript": f"// JavaScript function: {description}\nfunction",
        "java": f"// Java method: {description}\npublic"
    }
    
    input_text = lang_prompts.get(language, description)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Use more deterministic generation for code
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=0.3,        # Lower temperature for more predictable code
        top_p=0.8,
        repetition_penalty=1.1
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test code generation
code_descriptions = [
    "calculate fibonacci numbers",
    "sort a list of numbers",
    "check if a string is palindrome"
]

for desc in code_descriptions:
    print(f"\n=== Python: {desc} ===")
    code = code_generator(desc, language="python")
    print(code)
```

---

## 6. Further Reading

**Foundational Papers:**
- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215) - Original seq2seq paper
- [A Neural Conversational Model (Vinyals & Le, 2015)](https://arxiv.org/abs/1506.05869) - Neural dialogue systems

**Advanced Topics:**
- **Controlled Generation:** Steering text generation with attributes
- **Few-Shot Learning:** Generating text with minimal examples
- **Evaluation Metrics:** Measuring text generation quality
- **Ethical Considerations:** Bias, safety, and responsible AI

**Next Steps:**
1. **Experiment with different models:** Try GPT-2, GPT-3, BERT, T5
2. **Learn prompt engineering:** Design better inputs for your use case
3. **Explore fine-tuning:** Adapt models to your specific domain
4. **Study evaluation metrics:** Understand how to measure generation quality

> **Explore these papers to learn more about the foundations and advances in text generation!**

**Practice Exercise:**
Build a simple text generation API:
```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        temperature=temperature
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({
        'prompt': prompt,
        'generated_text': generated_text,
        'parameters': {
            'max_length': max_length,
            'temperature': temperature
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
``` 