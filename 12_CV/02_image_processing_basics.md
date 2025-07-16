# Image Processing Basics

Image processing is the foundation of computer vision, involving techniques to enhance, filter, and analyze digital images. This guide covers fundamental image processing operations and their mathematical foundations.

## Table of Contents

1. [Filtering and Enhancement](#filtering-and-enhancement)
2. [Edge Detection](#edge-detection)
3. [Morphological Operations](#morphological-operations)
4. [Histogram Processing](#histogram-processing)
5. [Noise Reduction](#noise-reduction)

## Filtering and Enhancement

### Linear Filters

Linear filters operate on pixels using a weighted sum of neighboring pixels. The most common linear filter is the Gaussian filter.

#### Gaussian Filter

The Gaussian filter is defined by the 2D Gaussian function:

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

def gaussian_filter_demo():
    # Create a test image with noise
    np.random.seed(42)
    original = np.zeros((100, 100))
    original[20:80, 20:80] = 255  # White square
    
    # Add noise
    noisy = original + np.random.normal(0, 30, original.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Apply Gaussian filter with different sigma values
    sigma_values = [0.5, 1.0, 2.0, 3.0]
    filtered_images = []
    
    for sigma in sigma_values:
        # Create Gaussian kernel
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        filtered = cv2.GaussianBlur(noisy, (kernel_size, kernel_size), sigma)
        filtered_images.append(filtered)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    for i, (sigma, filtered) in enumerate(zip(sigma_values, filtered_images)):
        row = (i + 2) // 3
        col = (i + 2) % 3
        axes[row, col].imshow(filtered, cmap='gray')
        axes[row, col].set_title(f'Gaussian Filter (σ={sigma})')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compare noise reduction
    print("Noise reduction comparison:")
    for sigma, filtered in zip(sigma_values, filtered_images):
        mse = np.mean((original.astype(float) - filtered.astype(float))**2)
        print(f"σ={sigma}: MSE = {mse:.2f}")

gaussian_filter_demo()
```

#### Mean Filter

The mean filter replaces each pixel with the average of its neighbors:

$$I'(x, y) = \frac{1}{N} \sum_{i=-k}^{k} \sum_{j=-k}^{k} I(x+i, y+j)$$

```python
def mean_filter_demo():
    # Create test image
    image = np.zeros((50, 50))
    image[10:40, 10:40] = 255
    image[20:30, 20:30] = 0  # Hole in the middle
    
    # Add salt and pepper noise
    noisy = image.copy()
    noise_pixels = np.random.choice([0, 255], size=image.shape, p=[0.1, 0.1])
    noisy = np.where(noise_pixels == 0, 0, np.where(noise_pixels == 255, 255, noisy))
    
    # Apply mean filter with different kernel sizes
    kernel_sizes = [3, 5, 7, 9]
    filtered_images = []
    
    for ksize in kernel_sizes:
        kernel = np.ones((ksize, ksize)) / (ksize * ksize)
        filtered = cv2.filter2D(noisy, -1, kernel)
        filtered_images.append(filtered)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy Image')
    axes[0, 1].axis('off')
    
    for i, (ksize, filtered) in enumerate(zip(kernel_sizes, filtered_images)):
        row = (i + 2) // 3
        col = (i + 2) % 3
        axes[row, col].imshow(filtered, cmap='gray')
        axes[row, col].set_title(f'Mean Filter ({ksize}x{ksize})')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

mean_filter_demo()
```

### Non-Linear Filters

Non-linear filters don't follow the principle of superposition and are often more effective for certain types of noise.

#### Median Filter

The median filter replaces each pixel with the median of its neighbors, effective for salt-and-pepper noise:

```python
def median_filter_demo():
    # Create test image
    image = np.zeros((50, 50))
    image[10:40, 10:40] = 255
    
    # Add salt and pepper noise
    noisy = image.copy()
    noise_mask = np.random.random(image.shape) < 0.1
    noisy[noise_mask] = np.random.choice([0, 255], size=noise_mask.sum())
    
    # Apply median filter with different kernel sizes
    kernel_sizes = [3, 5, 7]
    filtered_images = []
    
    for ksize in kernel_sizes:
        filtered = cv2.medianBlur(noisy, ksize)
        filtered_images.append(filtered)
    
    # Compare with mean filter
    mean_filtered = cv2.blur(noisy, (5, 5))
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Salt & Pepper Noise')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(mean_filtered, cmap='gray')
    axes[0, 2].set_title('Mean Filter (5x5)')
    axes[0, 2].axis('off')
    
    for i, (ksize, filtered) in enumerate(zip(kernel_sizes, filtered_images)):
        axes[1, i].imshow(filtered, cmap='gray')
        axes[1, i].set_title(f'Median Filter ({ksize}x{ksize})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compare performance
    print("Filter performance comparison:")
    print(f"Mean filter MSE: {np.mean((image - mean_filtered)**2):.2f}")
    for ksize, filtered in zip(kernel_sizes, filtered_images):
        mse = np.mean((image - filtered)**2)
        print(f"Median filter ({ksize}x{ksize}) MSE: {mse:.2f}")

median_filter_demo()
```

## Edge Detection

Edge detection identifies boundaries between different regions in an image. The most common methods are based on gradient operators.

### Gradient Operators

#### Sobel Operator

The Sobel operator computes gradients using two 3×3 kernels:

$$G_x = \begin{pmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{pmatrix}, \quad G_y = \begin{pmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{pmatrix}$$

The gradient magnitude is: $G = \sqrt{G_x^2 + G_y^2}$

```python
def sobel_edge_detection():
    # Create test image with edges
    image = np.zeros((100, 100))
    image[20:80, 20:80] = 255  # Square
    image[40:60, 40:60] = 0    # Hole
    
    # Add some noise
    noisy = image + np.random.normal(0, 10, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Apply Sobel operator
    sobelx = cv2.Sobel(noisy, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(noisy, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx)
    
    # Normalize for display
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(np.abs(sobelx), cmap='gray')
    axes[0, 1].set_title('Sobel X')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(sobely), cmap='gray')
    axes[0, 2].set_title('Sobel Y')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(magnitude_norm, cmap='gray')
    axes[1, 0].set_title('Gradient Magnitude')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(direction, cmap='hsv')
    axes[1, 1].set_title('Gradient Direction')
    axes[1, 1].axis('off')
    
    # Apply threshold to magnitude
    threshold = 50
    edges = (magnitude_norm > threshold).astype(np.uint8) * 255
    axes[1, 2].imshow(edges, cmap='gray')
    axes[1, 2].set_title(f'Edges (threshold={threshold})')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Gradient magnitude range: {magnitude.min():.2f} - {magnitude.max():.2f}")
    print(f"Gradient direction range: {direction.min():.2f} - {direction.max():.2f}")

sobel_edge_detection()
```

#### Laplacian Operator

The Laplacian operator is a second-order derivative operator:

$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

```python
def laplacian_edge_detection():
    # Create test image
    image = np.zeros((100, 100))
    image[20:80, 20:80] = 255
    
    # Add noise
    noisy = image + np.random.normal(0, 15, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(noisy, cv2.CV_64F)
    
    # Apply Gaussian smoothing before Laplacian (LoG - Laplacian of Gaussian)
    blurred = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    log = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Normalize for display
    laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    log_norm = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(laplacian_norm, cmap='gray')
    axes[0, 1].set_title('Laplacian')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(blurred, cmap='gray')
    axes[1, 0].set_title('Gaussian Blurred')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(log_norm, cmap='gray')
    axes[1, 1].set_title('Laplacian of Gaussian (LoG)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Laplacian range: {laplacian.min():.2f} - {laplacian.max():.2f}")
    print(f"LoG range: {log.min():.2f} - {log.max():.2f}")

laplacian_edge_detection()
```

### Canny Edge Detection

Canny edge detection is a multi-stage algorithm that produces high-quality edges:

1. **Gaussian smoothing**
2. **Gradient computation**
3. **Non-maximum suppression**
4. **Double thresholding**
5. **Edge tracking**

```python
def canny_edge_detection():
    # Create test image
    image = np.zeros((100, 100))
    image[20:80, 20:80] = 255
    image[40:60, 40:60] = 0
    
    # Add noise
    noisy = image + np.random.normal(0, 20, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Apply Canny edge detection with different parameters
    thresholds = [(50, 150), (100, 200), (30, 100)]
    canny_results = []
    
    for low, high in thresholds:
        edges = cv2.Canny(noisy, low, high)
        canny_results.append(edges)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show intermediate steps for first threshold
    # Gaussian blur
    blurred = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title('Gaussian Blur')
    axes[0, 1].axis('off')
    
    # Sobel gradients
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    axes[0, 2].imshow(magnitude_norm, cmap='gray')
    axes[0, 2].set_title('Gradient Magnitude')
    axes[0, 2].axis('off')
    
    # Canny results
    for i, (edges, (low, high)) in enumerate(zip(canny_results, thresholds)):
        axes[1, i].imshow(edges, cmap='gray')
        axes[1, i].set_title(f'Canny ({low}, {high})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Compare edge detection methods
    print("Edge detection comparison:")
    sobel_edges = cv2.Sobel(noisy, cv2.CV_64F, 1, 0, ksize=3)
    sobel_edges = np.abs(sobel_edges)
    sobel_edges = (sobel_edges > 50).astype(np.uint8) * 255
    
    print(f"Sobel edges: {np.sum(sobel_edges > 0)} pixels")
    for i, (edges, (low, high)) in enumerate(zip(canny_results, thresholds)):
        print(f"Canny ({low}, {high}): {np.sum(edges > 0)} pixels")

canny_edge_detection()
```

## Morphological Operations

Morphological operations are based on set theory and are used for shape analysis and noise removal.

### Basic Operations

#### Erosion and Dilation

**Erosion** shrinks objects: $A \ominus B = \{z | B_z \subseteq A\}$

**Dilation** expands objects: $A \oplus B = \{z | \hat{B}_z \cap A \neq \emptyset\}$

```python
def morphological_operations_demo():
    # Create test image
    image = np.zeros((100, 100))
    image[20:80, 20:80] = 255  # Square
    image[30:70, 30:70] = 0    # Hole
    
    # Add noise
    noisy = image.copy()
    noise_pixels = np.random.choice([0, 255], size=image.shape, p=[0.05, 0.05])
    noisy = np.where(noise_pixels == 0, 0, np.where(noise_pixels == 255, 255, noisy))
    
    # Define structuring elements
    kernel_sizes = [3, 5, 7]
    operations = []
    
    for ksize in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        
        # Erosion
        eroded = cv2.erode(noisy, kernel)
        
        # Dilation
        dilated = cv2.dilate(noisy, kernel)
        
        # Opening (erosion followed by dilation)
        opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)
        
        # Closing (dilation followed by erosion)
        closed = cv2.morphologyEx(noisy, cv2.MORPH_CLOSE, kernel)
        
        operations.append({
            'kernel_size': ksize,
            'eroded': eroded,
            'dilated': dilated,
            'opened': opened,
            'closed': closed
        })
    
    # Display results
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    # Original and noisy
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy')
    axes[0, 1].axis('off')
    
    # Empty plots for spacing
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')
    
    # Operation results
    for i, op in enumerate(operations):
        ksize = op['kernel_size']
        
        axes[i+1, 0].imshow(op['eroded'], cmap='gray')
        axes[i+1, 0].set_title(f'Erosion ({ksize}x{ksize})')
        axes[i+1, 0].axis('off')
        
        axes[i+1, 1].imshow(op['dilated'], cmap='gray')
        axes[i+1, 1].set_title(f'Dilation ({ksize}x{ksize})')
        axes[i+1, 1].axis('off')
        
        axes[i+1, 2].imshow(op['opened'], cmap='gray')
        axes[i+1, 2].set_title(f'Opening ({ksize}x{ksize})')
        axes[i+1, 2].axis('off')
        
        axes[i+1, 3].imshow(op['closed'], cmap='gray')
        axes[i+1, 3].set_title(f'Closing ({ksize}x{ksize})')
        axes[i+1, 3].axis('off')
        
        # Morphological gradient
        gradient = op['dilated'] - op['eroded']
        axes[i+1, 4].imshow(gradient, cmap='gray')
        axes[i+1, 4].set_title(f'Gradient ({ksize}x{ksize})')
        axes[i+1, 4].axis('off')
    
    plt.tight_layout()
    plt.show()

morphological_operations_demo()
```

## Histogram Processing

Histogram processing techniques modify the distribution of pixel intensities to enhance image contrast and visibility.

### Histogram Equalization

Histogram equalization spreads the pixel intensities across the full range:

$$s_k = T(r_k) = \sum_{j=0}^{k} \frac{n_j}{n}$$

where $n_j$ is the number of pixels with intensity $j$ and $n$ is the total number of pixels.

```python
def histogram_processing_demo():
    # Create test images with different contrast
    np.random.seed(42)
    
    # Low contrast image
    low_contrast = np.random.normal(128, 20, (100, 100))
    low_contrast = np.clip(low_contrast, 0, 255).astype(np.uint8)
    
    # High contrast image
    high_contrast = np.random.normal(128, 60, (100, 100))
    high_contrast = np.clip(high_contrast, 0, 255).astype(np.uint8)
    
    # Uneven histogram image
    uneven = np.zeros((100, 100), dtype=np.uint8)
    uneven[:50, :] = np.random.randint(0, 50, (50, 100))
    uneven[50:, :] = np.random.randint(200, 255, (50, 100))
    
    images = [low_contrast, high_contrast, uneven]
    titles = ['Low Contrast', 'High Contrast', 'Uneven Histogram']
    
    # Apply histogram equalization
    equalized_images = []
    for img in images:
        equalized = cv2.equalizeHist(img)
        equalized_images.append(equalized)
    
    # Display results
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    for i, (img, eq_img, title) in enumerate(zip(images, equalized_images, titles)):
        # Original image
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'{title} - Original')
        axes[i, 0].axis('off')
        
        # Equalized image
        axes[i, 1].imshow(eq_img, cmap='gray')
        axes[i, 1].set_title(f'{title} - Equalized')
        axes[i, 1].axis('off')
        
        # Original histogram
        axes[i, 2].hist(img.ravel(), bins=50, alpha=0.7, color='blue')
        axes[i, 2].set_title('Original Histogram')
        axes[i, 2].set_xlabel('Pixel Value')
        axes[i, 2].set_ylabel('Frequency')
        
        # Equalized histogram
        axes[i, 3].hist(eq_img.ravel(), bins=50, alpha=0.7, color='red')
        axes[i, 3].set_title('Equalized Histogram')
        axes[i, 3].set_xlabel('Pixel Value')
        axes[i, 3].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    print("Histogram statistics:")
    for i, (img, eq_img, title) in enumerate(zip(images, equalized_images, titles)):
        print(f"\n{title}:")
        print(f"  Original - Mean: {img.mean():.1f}, Std: {img.std():.1f}")
        print(f"  Equalized - Mean: {eq_img.mean():.1f}, Std: {eq_img.std():.1f}")

histogram_processing_demo()
```

### Adaptive Histogram Equalization (CLAHE)

CLAHE improves local contrast by applying histogram equalization to small regions:

```python
def clahe_demo():
    # Create test image with varying contrast
    image = np.zeros((200, 200), dtype=np.uint8)
    
    # Create gradient pattern
    for i in range(200):
        for j in range(200):
            # Varying contrast across the image
            contrast = 50 + 100 * (i / 200)
            image[i, j] = np.clip(128 + contrast * np.sin(j / 20), 0, 255)
    
    # Add some noise
    noisy = image + np.random.normal(0, 10, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Apply different histogram equalization methods
    # Global histogram equalization
    global_eq = cv2.equalizeHist(noisy)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_eq = clahe.apply(noisy)
    
    # CLAHE with different parameters
    clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    clahe_eq2 = clahe2.apply(noisy)
    
    # Display results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original images
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(global_eq, cmap='gray')
    axes[0, 1].set_title('Global Equalization')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(clahe_eq, cmap='gray')
    axes[0, 2].set_title('CLAHE (clip=2.0, tile=8x8)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(clahe_eq2, cmap='gray')
    axes[0, 3].set_title('CLAHE (clip=4.0, tile=16x16)')
    axes[0, 3].axis('off')
    
    # Histograms
    axes[1, 0].hist(noisy.ravel(), bins=50, alpha=0.7)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Pixel Value')
    
    axes[1, 1].hist(global_eq.ravel(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Global Equalized Histogram')
    axes[1, 1].set_xlabel('Pixel Value')
    
    axes[1, 2].hist(clahe_eq.ravel(), bins=50, alpha=0.7)
    axes[1, 2].set_title('CLAHE Histogram')
    axes[1, 2].set_xlabel('Pixel Value')
    
    axes[1, 3].hist(clahe_eq2.ravel(), bins=50, alpha=0.7)
    axes[1, 3].set_title('CLAHE (High Clip) Histogram')
    axes[1, 3].set_xlabel('Pixel Value')
    
    plt.tight_layout()
    plt.show()
    
    # Compare local contrast
    def local_contrast(img, window_size=20):
        """Calculate local contrast using standard deviation"""
        contrast_map = np.zeros_like(img, dtype=float)
        pad = window_size // 2
        
        for i in range(pad, img.shape[0] - pad):
            for j in range(pad, img.shape[1] - pad):
                window = img[i-pad:i+pad+1, j-pad:j+pad+1]
                contrast_map[i, j] = window.std()
        
        return contrast_map
    
    print("Local contrast comparison:")
    methods = [noisy, global_eq, clahe_eq, clahe_eq2]
    method_names = ['Original', 'Global EQ', 'CLAHE', 'CLAHE (High Clip)']
    
    for img, name in zip(methods, method_names):
        contrast = local_contrast(img)
        print(f"{name}: Mean local contrast = {contrast.mean():.2f}")

clahe_demo()
```

## Noise Reduction

### Advanced Noise Reduction Techniques

#### Bilateral Filter

The bilateral filter preserves edges while smoothing:

$$BF[I]_p = \frac{1}{W_p} \sum_{q \in S} G_{\sigma_s}(\|p-q\|) G_{\sigma_r}(|I_p - I_q|) I_q$$

where $G_{\sigma_s}$ and $G_{\sigma_r}$ are spatial and range Gaussian kernels.

```python
def bilateral_filter_demo():
    # Create test image
    image = np.zeros((100, 100))
    image[20:80, 20:80] = 255
    image[40:60, 40:60] = 0
    
    # Add Gaussian noise
    noisy = image + np.random.normal(0, 25, image.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Apply different filters
    # Gaussian filter
    gaussian_filtered = cv2.GaussianBlur(noisy, (9, 9), 2.0)
    
    # Bilateral filter
    bilateral_filtered = cv2.bilateralFilter(noisy, 9, 75, 75)
    
    # Non-local means
    nlm_filtered = cv2.fastNlMeansDenoising(noisy)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('Noisy')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gaussian_filtered, cmap='gray')
    axes[0, 2].set_title('Gaussian Filter')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(bilateral_filtered, cmap='gray')
    axes[1, 0].set_title('Bilateral Filter')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(nlm_filtered, cmap='gray')
    axes[1, 1].set_title('Non-local Means')
    axes[1, 1].axis('off')
    
    # Edge preservation comparison
    def edge_preservation(original, filtered):
        """Calculate edge preservation metric"""
        # Sobel edges
        sobel_orig = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
        sobel_filt = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
        
        # Correlation between edge maps
        correlation = np.corrcoef(sobel_orig.ravel(), sobel_filt.ravel())[0, 1]
        return correlation
    
    # Calculate metrics
    methods = [gaussian_filtered, bilateral_filtered, nlm_filtered]
    method_names = ['Gaussian', 'Bilateral', 'Non-local Means']
    
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.8, 'Edge Preservation Metrics:', fontsize=12, fontweight='bold')
    
    y_pos = 0.7
    for method, name in zip(methods, method_names):
        mse = np.mean((image.astype(float) - method.astype(float))**2)
        edge_pres = edge_preservation(image, method)
        
        axes[1, 2].text(0.1, y_pos, f'{name}:', fontsize=10, fontweight='bold')
        axes[1, 2].text(0.1, y_pos-0.05, f'  MSE: {mse:.1f}', fontsize=9)
        axes[1, 2].text(0.1, y_pos-0.1, f'  Edge Pres: {edge_pres:.3f}', fontsize=9)
        y_pos -= 0.2
    
    plt.tight_layout()
    plt.show()

bilateral_filter_demo()
```

## Summary

This guide covered fundamental image processing techniques:

1. **Filtering and Enhancement**: Linear and non-linear filters for noise reduction
2. **Edge Detection**: Gradient-based operators and advanced methods like Canny
3. **Morphological Operations**: Shape-based processing using erosion, dilation, opening, and closing
4. **Histogram Processing**: Contrast enhancement through histogram equalization
5. **Advanced Noise Reduction**: Bilateral filtering and non-local means

### Key Takeaways

- **Linear filters** are computationally efficient but may blur edges
- **Non-linear filters** like median and bilateral preserve edges better
- **Edge detection** requires careful parameter tuning for optimal results
- **Morphological operations** are powerful for shape analysis and noise removal
- **Histogram processing** can significantly improve image visibility
- **Advanced techniques** like CLAHE and bilateral filtering provide better results for specific applications

### Next Steps

With these basics mastered, you can explore:
- Feature detection and description algorithms
- Object detection and recognition
- Image segmentation techniques
- Deep learning approaches in computer vision 