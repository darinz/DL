# Multimodal Models

Multimodal models are designed to process and relate information from multiple modalities, such as text, images, and audio. These models learn joint representations that enable tasks like image captioning, visual question answering, and text-to-image generation.

---

## 1. What is a Multimodal Model?

A multimodal model integrates information from different sources (modalities). For example, CLIP learns to align images and their textual descriptions in a shared embedding space.

> **Note:** A modality is a type of data, such as text, image, audio, or video. Multimodal models can process two or more modalities together.

---

## 2. Joint Embedding Space

The core idea is to map different modalities into a common space where semantically similar items are close together.

```math
\text{Similarity}(x, y) = \frac{f(x) \cdot g(y)}{\|f(x)\| \|g(y)\|}
```

Where:
- $f(x)$ = embedding of image $x$
- $g(y)$ = embedding of text $y$

**Explanation:**
- $f(x)$ and $g(y)$ are functions (usually neural networks) that map images and text into vectors (embeddings) in the same space.
- The numerator $f(x) \cdot g(y)$ is the dot product, measuring how aligned the two vectors are.
- The denominator $\|f(x)\| \|g(y)\|$ normalizes the vectors, so the similarity is the cosine of the angle between them (cosine similarity), ranging from -1 (opposite) to 1 (identical).
- High similarity means the image and text are likely to match semantically.

---

## 3. Example: CLIP (Contrastive Language–Image Pretraining)

CLIP is trained to maximize the similarity between correct image-text pairs and minimize it for incorrect pairs using a contrastive loss.

### Contrastive Loss

```math
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f(x), g(y))/\tau)}{\sum_{y'} \exp(\text{sim}(f(x), g(y'))/\tau)}
```

Where $\tau$ is a temperature parameter.

**Explanation:**
- $\text{sim}(f(x), g(y))$ is the similarity (e.g., cosine similarity) between the image and text embeddings.
- The numerator is the exponentiated similarity for the correct image-text pair.
- The denominator sums over all possible text descriptions $y'$, including incorrect ones.
- $\tau$ (temperature) controls the sharpness of the distribution; lower $\tau$ makes the model focus more on the highest similarities.
- The loss encourages the model to assign higher similarity to correct pairs than to incorrect ones.

---

### Python Example: Using CLIP

```python
import torch
import clip
from PIL import Image

# Load model
model, preprocess = clip.load("ViT-B/32")  # Loads the CLIP model and preprocessing pipeline

# Prepare image and text
image = preprocess(Image.open("example.jpg")).unsqueeze(0)  # Preprocess and add batch dimension
text = clip.tokenize(["a photo of a cat", "a photo of a dog"])  # Tokenize text prompts

# Compute features
with torch.no_grad():
    image_features = model.encode_image(image)  # Get image embedding
    text_features = model.encode_text(text)     # Get text embeddings

    # Compute similarity
    logits_per_image, logits_per_text = model(image, text)  # Similarity scores
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # Convert to probabilities

print("Label probabilities:", probs)
```

**Code Annotations:**
- `clip.load` loads a pre-trained CLIP model and its preprocessing function.
- The image is preprocessed (resized, normalized) and converted to a batch tensor.
- `clip.tokenize` converts text prompts into token IDs for the model.
- `model.encode_image` and `model.encode_text` produce embeddings for the image and text, respectively.
- `model(image, text)` computes similarity logits between the image and each text prompt.
- `softmax` converts logits to probabilities, indicating how likely each text matches the image.

> **Tip:** You can extend this example to more images or text prompts for retrieval or classification tasks.

---

## 4. Applications
- **Image captioning:** Generating descriptive text for images.
- **Visual question answering:** Answering questions about images using both visual and textual information.
- **Text-to-image generation (e.g., DALL-E):** Creating images from textual descriptions.
- **Multimodal retrieval:** Finding images from text queries or vice versa.

---

## 5. Further Reading
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [DALL-E: Zero-Shot Text-to-Image Generation (Ramesh et al., 2021)](https://arxiv.org/abs/2102.12092) 

> **Explore these papers to learn more about how multimodal models are built and applied!** 