# Multimodal Models

Multimodal models are designed to process and relate information from multiple modalities, such as text, images, and audio. These models learn joint representations that enable tasks like image captioning, visual question answering, and text-to-image generation.

> **Learning Objective:** By the end of this guide, you'll understand how multimodal models work, the concept of joint embedding spaces, contrastive learning, and how to use models like CLIP for real-world applications.

---

## 1. What is a Multimodal Model?

A multimodal model integrates information from different sources (modalities). For example, CLIP learns to align images and their textual descriptions in a shared embedding space.

> **Key Insight:** A modality is a type of data, such as text, image, audio, or video. Multimodal models can process two or more modalities together.

**Deep Dive: Why Multimodal?**
- **Rich Information:** Different modalities provide complementary information
- **Real-World Applications:** Most real-world data is multimodal (videos have audio + visual)
- **Better Understanding:** Combining modalities often leads to better performance
- **Transfer Learning:** Knowledge from one modality can help another

**Common Modality Combinations:**
- **Text + Image:** CLIP, DALL-E, Stable Diffusion
- **Text + Audio:** Speech recognition, audio captioning
- **Image + Audio:** Video understanding
- **Text + Image + Audio:** Full video understanding

**Common Pitfall:** Don't assume more modalities always help - they need to be properly aligned and relevant to the task.

---

## 2. Joint Embedding Space

The core idea is to map different modalities into a common space where semantically similar items are close together.

```math
\text{Similarity}(x, y) = \frac{f(x) \cdot g(y)}{\|f(x)\| \|g(y)\|}
```

Where:
- $f(x)$ = embedding of image $x$
- $g(y)$ = embedding of text $y$

**Step-by-Step Math Breakdown:**

1. **Embedding Functions:** Each modality has its own encoder:
   ```math
   f: \text{Images} \rightarrow \mathbb{R}^d, \quad g: \text{Text} \rightarrow \mathbb{R}^d
   ```

2. **Dot Product:** Measures alignment between vectors:
   ```math
   f(x) \cdot g(y) = \sum_{i=1}^d f_i(x) g_i(y)
   ```

3. **Normalization:** Convert to cosine similarity:
   ```math
   \cos(\theta) = \frac{f(x) \cdot g(y)}{\|f(x)\| \|g(y)\|}
   ```

**Why Cosine Similarity?**
- **Scale Invariant:** Magnitude doesn't matter, only direction
- **Bounded Range:** Always between -1 and 1
- **Intuitive:** 1 = identical, 0 = orthogonal, -1 = opposite
- **Robust:** Less sensitive to outliers than Euclidean distance

**Intuitive Understanding:**
Think of embeddings as arrows in space. Similar items point in similar directions, so their cosine similarity is high. The dot product measures how much they point in the same direction, and normalization accounts for their lengths.

**Implementation Example:**
```python
import torch
import torch.nn.functional as F

def cosine_similarity(img_embedding, text_embedding):
    """
    Compute cosine similarity between image and text embeddings
    
    Args:
        img_embedding: Image embedding vector (batch_size, embedding_dim)
        text_embedding: Text embedding vector (batch_size, embedding_dim)
    
    Returns:
        similarity: Cosine similarity scores (batch_size,)
    """
    # Normalize embeddings to unit length
    img_norm = F.normalize(img_embedding, p=2, dim=1)
    text_norm = F.normalize(text_embedding, p=2, dim=1)
    
    # Compute cosine similarity
    similarity = torch.sum(img_norm * text_norm, dim=1)
    return similarity

# Example usage
img_emb = torch.randn(4, 512)  # 4 images, 512-dimensional embeddings
text_emb = torch.randn(4, 512)  # 4 text descriptions
similarities = cosine_similarity(img_emb, text_emb)
print(f"Similarities: {similarities}")
```

---

## 3. Example: CLIP (Contrastive Language–Image Pretraining)

CLIP is trained to maximize the similarity between correct image-text pairs and minimize it for incorrect pairs using a contrastive loss.

### Contrastive Loss

```math
\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(f(x), g(y))/\tau)}{\sum_{y'} \exp(\text{sim}(f(x), g(y'))/\tau)}
```

Where $\tau$ is a temperature parameter.

**Mathematical Intuition:**

1. **Numerator:** Probability of correct pairing
   ```math
   P(\text{correct}) = \frac{\exp(\text{sim}(f(x), g(y))/\tau)}{\sum_{y'} \exp(\text{sim}(f(x), g(y'))/\tau)}
   ```

2. **Denominator:** Sum over all possible text descriptions
   ```math
   \sum_{y'} \exp(\text{sim}(f(x), g(y'))/\tau)
   ```

3. **Loss:** Negative log-likelihood of correct pairing
   ```math
   \mathcal{L} = -\log P(\text{correct})
   ```

**Temperature Parameter (τ):**
- **Low τ (e.g., 0.1):** Sharp distribution, model is very confident
- **High τ (e.g., 1.0):** Soft distribution, model is more uncertain
- **Typical Value:** 0.07 for CLIP

**Why Contrastive Learning Works:**
- **Relative Learning:** Model learns what's similar vs. different
- **No Manual Labels:** Uses natural image-text pairs
- **Scalable:** Can use massive amounts of web data
- **Transferable:** Learned representations work for many tasks

**Implementation Details:**
```python
def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    Compute contrastive loss for image-text pairs
    
    Args:
        image_features: Image embeddings (batch_size, embedding_dim)
        text_features: Text embeddings (batch_size, embedding_dim)
        temperature: Temperature parameter for softmax
    
    Returns:
        loss: Contrastive loss value
    """
    # Normalize features
    image_features = F.normalize(image_features, p=2, dim=1)
    text_features = F.normalize(text_features, p=2, dim=1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.T) / temperature
    
    # Labels are diagonal (correct pairs)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    # Compute loss (cross-entropy with logits)
    loss_i = F.cross_entropy(logits, labels)  # Image to text
    loss_t = F.cross_entropy(logits.T, labels)  # Text to image
    
    return (loss_i + loss_t) / 2

# Example usage
batch_size = 32
embedding_dim = 512
img_feats = torch.randn(batch_size, embedding_dim)
txt_feats = torch.randn(batch_size, embedding_dim)
loss = contrastive_loss(img_feats, txt_feats)
print(f"Contrastive Loss: {loss.item():.4f}")
```

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

**Detailed Code Walkthrough:**

```python
# Step 1: Import and Setup
import torch
import clip
from PIL import Image
import numpy as np

# Step 2: Load CLIP Model
# Available models: "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
model, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")

# Step 3: Prepare Input Data
# Load and preprocess image
image_path = "example.jpg"
image = Image.open(image_path)
# preprocess: resizes to 224x224, normalizes pixel values, converts to tensor
processed_image = preprocess(image).unsqueeze(0)  # Add batch dimension

# Prepare text prompts
text_prompts = [
    "a photo of a cat",
    "a photo of a dog", 
    "a photo of a bird",
    "a photo of a car"
]
# tokenize: converts text to token IDs that the model can understand
tokenized_text = clip.tokenize(text_prompts)

# Step 4: Encode Features
with torch.no_grad():  # Disable gradient computation for inference
    # Encode image to embedding vector
    image_features = model.encode_image(processed_image)
    # Encode text to embedding vectors
    text_features = model.encode_text(tokenized_text)
    
    # Normalize features for cosine similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Step 5: Compute Similarities
# Compute cosine similarity between image and each text prompt
similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
probabilities = similarities.cpu().numpy()

# Step 6: Display Results
for i, (prompt, prob) in enumerate(zip(text_prompts, probabilities[0])):
    print(f"{prompt}: {prob:.3f}")

# Find best match
best_match_idx = np.argmax(probabilities[0])
print(f"\nBest match: {text_prompts[best_match_idx]} ({probabilities[0][best_match_idx]:.3f})")
```

**Try It Yourself:**
```python
# Experiment with different image-text pairs
def analyze_image_text_similarity(image_path, text_prompts):
    """Analyze similarity between an image and multiple text descriptions"""
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    tokenized_text = clip.tokenize(text_prompts)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokenized_text)
        
        # Normalize and compute similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    return similarities.cpu().numpy()[0]

# Test with your own images and prompts
test_prompts = [
    "a beautiful sunset",
    "a busy city street", 
    "a peaceful forest",
    "a modern building"
]
probs = analyze_image_text_similarity("your_image.jpg", test_prompts)
for prompt, prob in zip(test_prompts, probs):
    print(f"{prompt}: {prob:.3f}")
```

**Advanced CLIP Usage:**
```python
# Zero-shot image classification
def zero_shot_classification(image_path, class_names):
    """Perform zero-shot classification using CLIP"""
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text_prompts = [f"a photo of a {class_name}" for class_name in class_names]
    tokenized_text = clip.tokenize(text_prompts)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokenized_text)
        
        # Compute similarities
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    return similarities.cpu().numpy()[0]

# Example: Classify an image into 5 categories
classes = ["cat", "dog", "bird", "car", "tree"]
probabilities = zero_shot_classification("test_image.jpg", classes)
for class_name, prob in zip(classes, probabilities):
    print(f"{class_name}: {prob:.3f}")
```

> **Pro Tip:** CLIP works best with descriptive prompts. Instead of "cat", try "a photo of a cat sitting on a windowsill" for better results.

---

## 4. Applications

**Image Captioning:**
- **Automatic Descriptions:** Generate text descriptions for images
- **Accessibility:** Help visually impaired users understand images
- **Content Moderation:** Identify inappropriate content

**Visual Question Answering:**
- **Interactive AI:** Answer questions about image content
- **Educational Tools:** Help students understand visual concepts
- **Customer Support:** Answer product-related questions

**Text-to-Image Generation:**
- **Creative Tools:** Generate images from text descriptions
- **Design Assistance:** Create visual content for marketing
- **Prototyping:** Quickly visualize concepts

**Multimodal Retrieval:**
- **Image Search:** Find images using text queries
- **Text Search:** Find relevant text using image queries
- **Cross-Modal Matching:** Align different types of content

**Example Application: Image Search**
```python
def image_search(query_text, image_database, top_k=5):
    """
    Search for images using text query
    
    Args:
        query_text: Text description to search for
        image_database: List of image paths
        top_k: Number of top results to return
    """
    # Encode query
    tokenized_query = clip.tokenize([query_text])
    with torch.no_grad():
        query_features = model.encode_text(tokenized_query)
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
    
    # Encode all images
    image_features_list = []
    for img_path in image_database:
        image = preprocess(Image.open(img_path)).unsqueeze(0)
        with torch.no_grad():
            img_features = model.encode_image(image)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            image_features_list.append(img_features)
    
    # Stack all image features
    all_image_features = torch.cat(image_features_list, dim=0)
    
    # Compute similarities
    similarities = (100.0 * query_features @ all_image_features.T).squeeze()
    
    # Get top-k results
    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    
    return [(image_database[i], similarities[i].item()) for i in top_indices]

# Example usage
database = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
results = image_search("a cute cat playing", database, top_k=3)
for img_path, score in results:
    print(f"{img_path}: {score:.2f}")
```

---

## 5. Further Reading

**Foundational Papers:**
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) - The original CLIP paper
- [DALL-E: Zero-Shot Text-to-Image Generation (Ramesh et al., 2021)](https://arxiv.org/abs/2102.12092) - Text-to-image generation

**Advanced Topics:**
- **Contrastive Learning:** Understanding the training objective
- **Vision Transformers:** How CLIP processes images
- **Prompt Engineering:** Designing effective text prompts
- **Fine-tuning:** Adapting CLIP to specific domains

**Next Steps:**
1. **Experiment with CLIP:** Try different prompts and images
2. **Explore Other Models:** Look into DALL-E, Stable Diffusion, Flamingo
3. **Learn Prompt Engineering:** Design better text descriptions
4. **Study Evaluation Metrics:** Understand how to measure multimodal performance

> **Explore these papers to learn more about how multimodal models are built and applied!**

**Practice Exercise:**
Build a simple image classifier using CLIP:
```python
def build_image_classifier(class_names):
    """Build a zero-shot classifier for given classes"""
    def classify_image(image_path):
        image = preprocess(Image.open(image_path)).unsqueeze(0)
        text_prompts = [f"a photo of a {name}" for name in class_names]
        tokenized_text = clip.tokenize(text_prompts)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(tokenized_text)
            
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        return similarities.cpu().numpy()[0]
    
    return classify_image

# Test your classifier
classifier = build_image_classifier(["cat", "dog", "bird", "car"])
probabilities = classifier("test_image.jpg")
for class_name, prob in zip(["cat", "dog", "bird", "car"], probabilities):
    print(f"{class_name}: {prob:.3f}")
``` 