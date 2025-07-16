# Multimodal Models

Multimodal models are designed to process and relate information from multiple modalities, such as text, images, and audio. These models learn joint representations that enable tasks like image captioning, visual question answering, and text-to-image generation.

## 1. What is a Multimodal Model?

A multimodal model integrates information from different sources (modalities). For example, CLIP learns to align images and their textual descriptions in a shared embedding space.

## 2. Joint Embedding Space

The core idea is to map different modalities into a common space where semantically similar items are close together.

```math
\text{Similarity}(x, y) = \frac{f(x) \cdot g(y)}{\|f(x)\| \|g(y)\|}
```

Where:
- $`f(x)`$ = embedding of image $`x`$
- $`g(y)`$ = embedding of text $`y`$

## 3. Example: CLIP (Contrastive Language–Image Pretraining)

CLIP is trained to maximize the similarity between correct image-text pairs and minimize it for incorrect pairs using a contrastive loss.

### Contrastive Loss

```math
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f(x), g(y))/\tau)}{\sum_{y'} \exp(\text{sim}(f(x), g(y'))/\tau)}
```

Where $`\tau`$ is a temperature parameter.

### Python Example: Using CLIP

```python
import torch
import clip
from PIL import Image

# Load model
model, preprocess = clip.load("ViT-B/32")

# Prepare image and text
image = preprocess(Image.open("example.jpg")).unsqueeze(0)
text = clip.tokenize(["a photo of a cat", "a photo of a dog"])

# Compute features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # Compute similarity
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probabilities:", probs)
```

## 4. Applications
- Image captioning
- Visual question answering
- Text-to-image generation (e.g., DALL-E)
- Multimodal retrieval

## 5. Further Reading
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [DALL-E: Zero-Shot Text-to-Image Generation (Ramesh et al., 2021)](https://arxiv.org/abs/2102.12092) 