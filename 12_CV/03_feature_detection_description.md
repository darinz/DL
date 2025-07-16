# Feature Detection and Description

## 1. Traditional Feature Detectors

### Harris Corner Detector

The Harris corner detector identifies corners by analyzing the local autocorrelation matrix.

**Autocorrelation Matrix:**
```math
M = \begin{bmatrix} \sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2 \end{bmatrix}
```

**Corner Response Function:**
```math
R = \det(M) - k \cdot \text{trace}(M)^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2
```

Where:
- $\lambda_1, \lambda_2$ are eigenvalues of $M$
- $k$ is a sensitivity parameter (typically 0.04-0.06)
- $I_x, I_y$ are image gradients

**Corner Classification:**
- $R > 0$ and large: Corner
- $R < 0$: Edge
- $R \approx 0$: Flat region

### SIFT (Scale-Invariant Feature Transform)

SIFT detects and describes features that are invariant to scale, rotation, and illumination changes.

#### Scale Space Construction
**Gaussian Scale Space:**
```math
L(x, y, \sigma) = G(x, y, \sigma) * I(x, y)
```

**Difference of Gaussians (DoG):**
```math
D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma)
```

Where $G(x, y, \sigma) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$

#### Keypoint Detection
1. **Extrema Detection:** Find local maxima/minima in DoG pyramid
2. **Keypoint Localization:** Refine keypoint location using Taylor expansion
3. **Orientation Assignment:** Compute dominant gradient orientation

**Gradient Magnitude and Orientation:**
```math
m(x, y) = \sqrt{L_x^2 + L_y^2}
```
```math
\theta(x, y) = \arctan\left(\frac{L_y}{L_x}\right)
```

#### SIFT Descriptor
**128-dimensional descriptor:**
- 4×4 spatial bins
- 8 orientation bins per spatial bin
- Total: 4 × 4 × 8 = 128 dimensions

**Descriptor Computation:**
```math
d_i = \sum_{(x,y) \in \text{bin}_i} m(x, y) \cdot w(x, y) \cdot \delta(\theta(x, y))
```

Where:
- $w(x, y)$ is the Gaussian weight
- $\delta(\theta)$ is the orientation bin assignment

### SURF (Speeded Up Robust Features)

SURF is a faster alternative to SIFT with similar performance.

#### Integral Image
**Definition:**
```math
I_{\Sigma}(x, y) = \sum_{i=0}^{x} \sum_{j=0}^{y} I(i, j)
```

**Box Filter Response:**
```math
D_{xx} = \sum_{i \in \text{box}} I(i) - 2 \sum_{i \in \text{center}} I(i) + \sum_{i \in \text{box}} I(i)
```

#### SURF Descriptor
**64-dimensional descriptor:**
- 4×4 sub-regions
- 4 responses per sub-region (dx, dy, |dx|, |dy|)
- Total: 4 × 4 × 4 = 64 dimensions

### ORB (Oriented FAST and Rotated BRIEF)

ORB combines FAST corner detection with rotated BRIEF descriptors.

#### FAST Corner Detection
**Corner Response:**
```math
\text{FAST}(p) = \begin{cases}
1 & \text{if } \sum_{i=1}^{16} |I(p_i) - I(p)| > \tau \\
0 & \text{otherwise}
\end{cases}
```

Where $p_i$ are the 16 pixels in a circle around $p$.

#### BRIEF Descriptor
**Binary descriptor:**
```math
\tau(p; x, y) = \begin{cases}
1 & \text{if } I(p + x) < I(p + y) \\
0 & \text{otherwise}
\end{cases}
```

**ORB Descriptor:**
```math
\text{ORB}(p) = \{\tau(p; x_i, y_i) : i = 1, 2, ..., 256\}
```

## 2. Feature Descriptors

### HOG (Histogram of Oriented Gradients)

HOG computes histograms of gradient orientations in local cells.

**Gradient Computation:**
```math
G_x = I(x+1, y) - I(x-1, y)
```
```math
G_y = I(x, y+1) - I(x, y-1)
```

**Gradient Magnitude and Orientation:**
```math
m(x, y) = \sqrt{G_x^2 + G_y^2}
```
```math
\theta(x, y) = \arctan\left(\frac{G_y}{G_x}\right)
```

**HOG Descriptor:**
```math
h_i = \sum_{(x,y) \in \text{cell}_i} m(x, y) \cdot \delta(\theta(x, y))
```

### LBP (Local Binary Pattern)

LBP encodes local texture information using binary patterns.

**LBP Operator:**
```math
\text{LBP}(x_c, y_c) = \sum_{i=0}^{7} 2^i \cdot s(I_i - I_c)
```

Where:
```math
s(x) = \begin{cases}
1 & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}
```

**Uniform LBP:**
```math
\text{LBP}_{u2}(x_c, y_c) = \begin{cases}
\sum_{i=0}^{7} s(I_i - I_c) & \text{if } U(\text{LBP}) \leq 2 \\
9 & \text{otherwise}
\end{cases}
```

Where $U(\text{LBP})$ is the number of bit transitions.

## 3. Deep Learning Features

### CNN Feature Extraction

Convolutional Neural Networks can extract powerful features from images.

**Convolutional Layer:**
```math
F_{i,j,k} = \sum_{m} \sum_{n} \sum_{c} I_{i+m, j+n, c} \cdot W_{m,n,c,k} + b_k
```

**Pooling Layer:**
```math
P_{i,j,k} = \max_{(m,n) \in R_{i,j}} F_{m,n,k}
```

### Transfer Learning for Features

**Feature Extraction:**
```math
\phi(x) = f_{L-1}(f_{L-2}(...f_1(x)))
```

Where $f_i$ are the layers of a pre-trained network.

## 4. Feature Matching

### Brute Force Matching

**Distance Metrics:**
- **Euclidean Distance:** $d(x, y) = \sqrt{\sum_{i} (x_i - y_i)^2}$
- **Manhattan Distance:** $d(x, y) = \sum_{i} |x_i - y_i|$
- **Cosine Distance:** $d(x, y) = 1 - \frac{x \cdot y}{\|x\| \|y\|}$

### FLANN (Fast Library for Approximate Nearest Neighbors)

**KD-Tree Search:**
```math
\text{Search}(q, T) = \arg\min_{p \in T} \|q - p\|
```

**LSH (Locality Sensitive Hashing):**
```math
h_i(x) = \left\lfloor \frac{a_i \cdot x + b_i}{w} \right\rfloor
```

Where $a_i$ is a random vector and $b_i$ is a random offset.

### RANSAC for Outlier Rejection

**RANSAC Algorithm:**
1. Randomly sample minimal subset
2. Fit model to subset
3. Count inliers
4. Repeat and keep best model

**Inlier Criterion:**
```math
\text{inlier} = \begin{cases}
1 & \text{if } d(x, \text{model}) < \tau \\
0 & \text{otherwise}
\end{cases}
```

## 5. Python Implementation Examples

### Traditional Feature Detectors

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d

# Create test images
def create_test_images():
    """Create test images for feature detection."""
    # Create a simple image with corners and edges
    img1 = np.zeros((200, 200), dtype=np.uint8)
    
    # Add geometric shapes
    cv2.rectangle(img1, (50, 50), (150, 150), 255, 2)
    cv2.circle(img1, (100, 100), 30, 255, 2)
    cv2.line(img1, (0, 0), (200, 200), 255, 2)
    
    # Create a transformed version
    img2 = img1.copy()
    # Apply rotation and scaling
    M = cv2.getRotationMatrix2D((100, 100), 30, 1.2)
    img2 = cv2.warpAffine(img2, M, (200, 200))
    
    # Add some noise
    noise1 = np.random.normal(0, 10, img1.shape).astype(np.uint8)
    noise2 = np.random.normal(0, 10, img2.shape).astype(np.uint8)
    
    img1 = cv2.add(img1, noise1)
    img2 = cv2.add(img2, noise2)
    
    return img1, img2

# Harris Corner Detector
def harris_corner_detector(image, k=0.04, threshold=0.01):
    """Implement Harris corner detector."""
    # Convert to float
    image = image.astype(np.float64)
    
    # Compute gradients
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute products of gradients
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy
    
    # Apply Gaussian smoothing
    kernel_size = 5
    Ixx = cv2.GaussianBlur(Ixx, (kernel_size, kernel_size), 1)
    Iyy = cv2.GaussianBlur(Iyy, (kernel_size, kernel_size), 1)
    Ixy = cv2.GaussianBlur(Ixy, (kernel_size, kernel_size), 1)
    
    # Compute corner response
    det_M = Ixx * Iyy - Ixy * Ixy
    trace_M = Ixx + Iyy
    R = det_M - k * trace_M * trace_M
    
    # Threshold and find local maxima
    corners = np.zeros_like(R)
    corners[R > threshold * R.max()] = 1
    
    # Non-maximum suppression
    corners = ndimage.maximum_filter(corners, size=5)
    corners = (corners == R) & (R > threshold * R.max())
    
    return corners, R

# SIFT-like feature detection
def sift_like_detector(image, num_octaves=4, num_scales=3):
    """Simplified SIFT-like feature detection."""
    # Create scale space
    scales = []
    sigmas = [1.6, 3.2, 6.4, 12.8]
    
    for sigma in sigmas:
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        scales.append(blurred)
    
    # Compute Difference of Gaussians
    dogs = []
    for i in range(len(scales) - 1):
        dog = scales[i+1] - scales[i]
        dogs.append(dog)
    
    # Find extrema in DoG
    keypoints = []
    for octave_idx, dog in enumerate(dogs):
        for i in range(1, dog.shape[0] - 1):
            for j in range(1, dog.shape[1] - 1):
                # Check if current pixel is extrema
                current = dog[i, j]
                neighbors = dog[i-1:i+2, j-1:j+2].flatten()
                
                if current == max(neighbors) or current == min(neighbors):
                    keypoints.append((j, i, octave_idx))
    
    return keypoints, dogs

# ORB feature detection
def orb_detector(image, max_features=500):
    """ORB feature detection using OpenCV."""
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=max_features)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    return keypoints, descriptors

# Feature matching demonstration
def demonstrate_feature_detection(img1, img2):
    """Demonstrate various feature detection methods."""
    # Harris corners
    corners1, response1 = harris_corner_detector(img1)
    corners2, response2 = harris_corner_detector(img2)
    
    # SIFT-like features
    keypoints1, dogs1 = sift_like_detector(img1)
    keypoints2, dogs2 = sift_like_detector(img2)
    
    # ORB features
    orb_kp1, orb_desc1 = orb_detector(img1)
    orb_kp2, orb_desc2 = orb_detector(img2)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title('Image 1')
    axes[0, 1].imshow(img2, cmap='gray')
    axes[0, 1].set_title('Image 2')
    
    # Harris corners
    axes[0, 2].imshow(img1, cmap='gray')
    axes[0, 2].imshow(corners1, alpha=0.7, cmap='hot')
    axes[0, 2].set_title('Harris Corners')
    
    # SIFT-like features
    axes[1, 0].imshow(img1, cmap='gray')
    for kp in keypoints1[:20]:  # Show first 20 keypoints
        axes[1, 0].plot(kp[0], kp[1], 'r+', markersize=10)
    axes[1, 0].set_title('SIFT-like Features')
    
    # ORB features
    axes[1, 1].imshow(img1, cmap='gray')
    for kp in orb_kp1[:20]:  # Show first 20 keypoints
        axes[1, 1].plot(kp.pt[0], kp.pt[1], 'g+', markersize=10)
    axes[1, 1].set_title('ORB Features')
    
    # Response maps
    axes[1, 2].imshow(response1, cmap='hot')
    axes[1, 2].set_title('Harris Response')
    
    plt.tight_layout()
    plt.show()
    
    return orb_kp1, orb_desc1, orb_kp2, orb_desc2

# Feature matching
def demonstrate_feature_matching(kp1, desc1, kp2, desc2, img1, img2):
    """Demonstrate feature matching techniques."""
    # Brute force matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # FLANN matching
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                       table_number=6,
                       key_size=12,
                       multi_probe_level=1)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann_matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match_pair in flann_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    # RANSAC for homography estimation
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Filter matches using RANSAC mask
        ransac_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    else:
        ransac_matches = good_matches
        H = None
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Brute force matches
    img_bf = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[0, 0].imshow(img_bf)
    axes[0, 0].set_title(f'Brute Force Matches ({len(matches)})')
    
    # FLANN matches
    img_flann = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[0, 1].imshow(img_flann)
    axes[0, 1].set_title(f'FLANN Matches ({len(good_matches)})')
    
    # RANSAC filtered matches
    img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, ransac_matches[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[1, 0].imshow(img_ransac)
    axes[1, 0].set_title(f'RANSAC Filtered ({len(ransac_matches)})')
    
    # Match statistics
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.8, f'Total matches: {len(matches)}', fontsize=12)
    axes[1, 1].text(0.1, 0.7, f'Good matches: {len(good_matches)}', fontsize=12)
    axes[1, 1].text(0.1, 0.6, f'RANSAC inliers: {len(ransac_matches)}', fontsize=12)
    if H is not None:
        axes[1, 1].text(0.1, 0.5, 'Homography found', fontsize=12)
    else:
        axes[1, 1].text(0.1, 0.5, 'No homography', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return matches, good_matches, ransac_matches, H

# HOG descriptor implementation
def compute_hog_descriptor(image, cell_size=8, block_size=2, num_bins=9):
    """Compute HOG descriptor for an image."""
    # Compute gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    
    # Compute magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi
    orientation[orientation < 0] += 180
    
    # Compute cell histograms
    h, w = image.shape
    cell_h, cell_w = cell_size, cell_size
    
    num_cells_h = h // cell_h
    num_cells_w = w // cell_w
    
    cell_histograms = np.zeros((num_cells_h, num_cells_w, num_bins))
    
    for i in range(num_cells_h):
        for j in range(num_cells_w):
            # Extract cell
            cell_mag = magnitude[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cell_ori = orientation[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            
            # Compute histogram
            for bin_idx in range(num_bins):
                bin_start = bin_idx * 180 / num_bins
                bin_end = (bin_idx + 1) * 180 / num_bins
                
                mask = (cell_ori >= bin_start) & (cell_ori < bin_end)
                cell_histograms[i, j, bin_idx] = np.sum(cell_mag[mask])
    
    # Block normalization
    block_h, block_w = block_size, block_size
    num_blocks_h = num_cells_h - block_h + 1
    num_blocks_w = num_cells_w - block_w + 1
    
    hog_descriptor = []
    
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            # Extract block
            block = cell_histograms[i:i+block_h, j:j+block_w, :]
            
            # Normalize block
            block_norm = np.sqrt(np.sum(block**2) + 1e-6)
            block_normalized = block / block_norm
            
            hog_descriptor.extend(block_normalized.flatten())
    
    return np.array(hog_descriptor)

# LBP descriptor implementation
def compute_lbp_descriptor(image, radius=1, num_points=8):
    """Compute LBP descriptor for an image."""
    h, w = image.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = image[i, j]
            code = 0
            
            for k in range(num_points):
                angle = 2 * np.pi * k / num_points
                x = int(i + radius * np.cos(angle))
                y = int(j + radius * np.sin(angle))
                
                if image[x, y] >= center:
                    code |= (1 << k)
            
            lbp[i, j] = code
    
    # Compute histogram
    num_bins = 2**num_points
    histogram = np.zeros(num_bins)
    
    for i in range(h):
        for j in range(w):
            histogram[lbp[i, j]] += 1
    
    # Normalize histogram
    histogram = histogram / np.sum(histogram)
    
    return histogram, lbp

# Deep learning feature extraction
def extract_cnn_features(image, model_name='vgg16'):
    """Extract CNN features using pre-trained models."""
    import tensorflow as tf
    from tensorflow.keras.applications import VGG16, ResNet50
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.vgg16 import preprocess_input
    
    # Load pre-trained model
    if model_name == 'vgg16':
        model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    elif model_name == 'resnet50':
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Preprocess image
    img_resized = cv2.resize(image, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    img_array = keras_image.img_to_array(img_rgb)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = model.predict(img_array)
    
    return features.flatten()

# Main demonstration
if __name__ == "__main__":
    # Create test images
    img1, img2 = create_test_images()
    
    # Demonstrate feature detection
    kp1, desc1, kp2, desc2 = demonstrate_feature_detection(img1, img2)
    
    # Demonstrate feature matching
    matches, good_matches, ransac_matches, H = demonstrate_feature_matching(
        kp1, desc1, kp2, desc2, img1, img2
    )
    
    # Demonstrate HOG descriptor
    hog_desc = compute_hog_descriptor(img1)
    print(f"HOG descriptor length: {len(hog_desc)}")
    
    # Demonstrate LBP descriptor
    lbp_hist, lbp_image = compute_lbp_descriptor(img1)
    print(f"LBP descriptor length: {len(lbp_hist)}")
    
    # Display LBP result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(lbp_image, cmap='gray')
    plt.title('LBP Image')
    plt.tight_layout()
    plt.show()
    
    # Plot HOG and LBP histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hog_desc)
    plt.title('HOG Descriptor')
    plt.xlabel('Feature Index')
    plt.ylabel('Value')
    plt.subplot(1, 2, 2)
    plt.plot(lbp_hist)
    plt.title('LBP Histogram')
    plt.xlabel('LBP Code')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
```

### Advanced Feature Analysis

```python
# Feature quality analysis
def analyze_feature_quality(kp1, desc1, kp2, desc2, matches):
    """Analyze the quality of detected features and matches."""
    # Match distance distribution
    distances = [m.distance for m in matches]
    
    # Keypoint distribution
    kp1_positions = np.array([kp.pt for kp in kp1])
    kp2_positions = np.array([kp.pt for kp in kp2])
    
    # Response strength
    responses1 = [kp.response for kp in kp1]
    responses2 = [kp.response for kp in kp2]
    
    # Display analysis
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Distance histogram
    axes[0, 0].hist(distances, bins=50, alpha=0.7)
    axes[0, 0].set_title('Match Distance Distribution')
    axes[0, 0].set_xlabel('Distance')
    axes[0, 0].set_ylabel('Frequency')
    
    # Keypoint distribution
    axes[0, 1].scatter(kp1_positions[:, 0], kp1_positions[:, 1], alpha=0.6)
    axes[0, 1].set_title('Keypoint Distribution (Image 1)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    
    axes[0, 2].scatter(kp2_positions[:, 0], kp2_positions[:, 1], alpha=0.6)
    axes[0, 2].set_title('Keypoint Distribution (Image 2)')
    axes[0, 2].set_xlabel('X')
    axes[0, 2].set_ylabel('Y')
    
    # Response strength
    axes[1, 0].hist(responses1, bins=50, alpha=0.7, label='Image 1')
    axes[1, 0].hist(responses2, bins=50, alpha=0.7, label='Image 2')
    axes[1, 0].set_title('Response Strength Distribution')
    axes[1, 0].set_xlabel('Response')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # Scale distribution
    scales1 = [kp.size for kp in kp1]
    scales2 = [kp.size for kp in kp2]
    axes[1, 1].hist(scales1, bins=50, alpha=0.7, label='Image 1')
    axes[1, 1].hist(scales2, bins=50, alpha=0.7, label='Image 2')
    axes[1, 1].set_title('Scale Distribution')
    axes[1, 1].set_xlabel('Scale')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    
    # Orientation distribution
    angles1 = [kp.angle for kp in kp1]
    angles2 = [kp.angle for kp in kp2]
    axes[1, 2].hist(angles1, bins=36, alpha=0.7, label='Image 1')
    axes[1, 2].hist(angles2, bins=36, alpha=0.7, label='Image 2')
    axes[1, 2].set_title('Orientation Distribution')
    axes[1, 2].set_xlabel('Angle (degrees)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Feature Analysis Statistics:")
    print(f"Number of keypoints (Image 1): {len(kp1)}")
    print(f"Number of keypoints (Image 2): {len(kp2)}")
    print(f"Number of matches: {len(matches)}")
    print(f"Average match distance: {np.mean(distances):.2f}")
    print(f"Match distance std: {np.std(distances):.2f}")

# Multi-scale feature detection
def multi_scale_feature_detection(image, scales=[0.5, 1.0, 2.0]):
    """Detect features at multiple scales."""
    all_keypoints = []
    all_descriptors = []
    
    for scale in scales:
        # Resize image
        h, w = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # Detect features
        orb = cv2.ORB_create(nfeatures=500)
        kp, desc = orb.detectAndCompute(resized, None)
        
        # Scale keypoint coordinates back to original size
        for keypoint in kp:
            keypoint.pt = (keypoint.pt[0] / scale, keypoint.pt[1] / scale)
            keypoint.size = keypoint.size / scale
        
        all_keypoints.extend(kp)
        if desc is not None:
            all_descriptors.extend(desc)
    
    return all_keypoints, np.array(all_descriptors)

# Feature tracking over time
def track_features_over_time(images):
    """Track features across multiple images."""
    # Initialize feature detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect features in first image
    kp1, desc1 = orb.detectAndCompute(images[0], None)
    
    tracks = []
    for i in range(1, len(images)):
        # Detect features in current image
        kp2, desc2 = orb.detectAndCompute(images[i], None)
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Estimate motion
        if len(matches) >= 4:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            tracks.append({
                'frame': i,
                'matches': len(matches),
                'inliers': np.sum(mask),
                'homography': H
            })
        
        # Update for next iteration
        kp1, desc1 = kp2, desc2
    
    return tracks
```

This comprehensive guide covers traditional and modern feature detection and description techniques. The mathematical foundations are essential for understanding how these algorithms work, while the Python implementations demonstrate practical applications in computer vision tasks. 