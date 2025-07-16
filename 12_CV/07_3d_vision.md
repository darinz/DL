# 3D Vision

> **Key Insight:** 3D vision enables machines to perceive, reconstruct, and reason about the three-dimensional world, powering robotics, AR/VR, autonomous vehicles, and more.

## 1. Overview

3D vision involves understanding and processing three-dimensional data, including point clouds, meshes, and volumetric representations. This field is crucial for robotics, autonomous vehicles, augmented reality, and computer graphics.

**Mathematical Representation:**
```math
P = \{p_i = (x_i, y_i, z_i) \in \mathbb{R}^3 : i = 1, 2, ..., N\}
```
Where $`P`$ is a point cloud with $`N`$ points in 3D space.

> **Did you know?**
> 3D vision is not just for perception—it's also used for simulation, digital twins, and virtual content creation.

---

## 2. Point Cloud Processing

### PointNet Architecture

PointNet is a deep learning architecture designed for point cloud processing. It learns features directly from unordered 3D points.

#### Architecture
**Input Transformation:**
$`T_1 = \text{MLP}(P) \in \mathbb{R}^{N \times 64}`$

**Feature Transformation:**
$`T_2 = \text{MLP}(T_1) \in \mathbb{R}^{N \times 1024}`$

**Global Feature:**
$`F_{global} = \max_{i} T_2(i) \in \mathbb{R}^{1024}`$

**Classification/Regression:**
$`Y = \text{MLP}(F_{global}) \in \mathbb{R}^{C}`$

#### Permutation Invariance
**Symmetric Function:**
$`f(\{x_1, ..., x_n\}) = \gamma \circ g(h(x_1), ..., h(x_n))`$
Where:
- $`h`$ is a shared MLP
- $`g`$ is a symmetric function (max pooling)
- $`\gamma`$ is a final MLP

> **Key Insight:**
> PointNet's use of symmetric functions (like max pooling) makes it invariant to the order of input points.

---

### PointNet++

PointNet++ extends PointNet with hierarchical feature learning, capturing local and global structures.

#### Hierarchical Sampling
**Farthest Point Sampling (FPS):**
$`p_{i+1} = \arg\max_{p \in P \setminus \{p_1, ..., p_i\}} \min_{j \leq i} \|p - p_j\|_2`$

#### Grouping
**Ball Query:**
$`N(p_i, r) = \{p_j : \|p_i - p_j\|_2 \leq r\}`$

**K-Nearest Neighbors:**
$`N(p_i, k) = \{p_j : j \in \text{top-k}(\|p_i - p_j\|_2)\}`$

#### Feature Aggregation
**Multi-scale Grouping:**
$`F_i = \text{concat}(F_i^1, F_i^2, ..., F_i^S)`$
Where $`F_i^s`$ is the feature at scale $`s`$.

> **Try it yourself!**
> Visualize the effect of different sampling and grouping strategies on a point cloud. How does local context affect feature learning?

---

## 3. Voxel-Based Methods

### VoxelNet

VoxelNet converts point clouds to voxels for 3D object detection.

#### Voxelization
**Point to Voxel Assignment:**
$`v_{ijk} = \{p \in P : \lfloor p_x/v \rfloor = i, \lfloor p_y/v \rfloor = j, \lfloor p_z/v \rfloor = k\}`$
Where $`v`$ is the voxel size.

#### Voxel Feature Encoding (VFE)
**VFE Layer:**
$`F_{out} = \max_{p \in v} \text{concat}(p, F_{in}(p))`$

#### Convolutional Middle Layers
**3D Convolution:**
$`F_{i+1} = \text{Conv3D}(F_i, W_i) + b_i`$

### PointPillars

PointPillars uses pillars (vertical columns) for efficient 3D detection.

#### Pillar Generation
**Pillar Assignment:**
$`p_{ij} = \{p \in P : \lfloor p_x/d_x \rfloor = i, \lfloor p_y/d_y \rfloor = j\}`$
Where $`d_x, d_y`$ are pillar dimensions.

#### Pillar Feature Net
**Feature Encoding:**
$`F_{pillar} = \text{PFN}(p_{ij}) \in \mathbb{R}^{C}`$

#### 2D Convolutional Backbone
**Pseudo-image:**
$`I_{pseudo} = \text{reshape}(F_{pillars}) \in \mathbb{R}^{H \times W \times C}`$

> **Common Pitfall:**
> Voxelization can lose fine geometric details. Always check the effect of voxel size on downstream tasks.

---

## 4. Multi-View Geometry

### Epipolar Geometry

#### Fundamental Matrix
**Epipolar Constraint:**
$`x_2^T F x_1 = 0`$

**Fundamental Matrix:**
$`F = K_2^{-T} [t]_{\times} R K_1^{-1}`$
Where $`[t]_{\times}`$ is the skew-symmetric matrix of translation vector.

#### Essential Matrix
**Essential Matrix:**
$`E = [t]_{\times} R`$

**Relationship:**
$`F = K_2^{-T} E K_1^{-1}`$

> **Geometric Intuition:**
> The fundamental and essential matrices encode the geometric relationship between two camera views, constraining where a point in one image can appear in the other.

---

### Structure from Motion (SfM)

#### Triangulation
**Linear Triangulation:**
```math
\begin{bmatrix}
x_1 p_1^3 - p_1^1 \\
y_1 p_1^3 - p_1^2 \\
x_2 p_2^3 - p_2^1 \\
y_2 p_2^3 - p_2^2
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix} = 0
```
Where $`p_i^j`$ is the $`j`$-th row of projection matrix $`P_i`$.

#### Bundle Adjustment
**Cost Function:**
$`\min_{P_i, X_j} \sum_{i,j} \|x_{ij} - P_i X_j\|_2^2`$

### SLAM (Simultaneous Localization and Mapping)

#### Visual SLAM
**Feature Matching:**
$`M_{ij} = \text{match}(f_i, f_j)`$

**Pose Estimation:**
$`T_{i+1} = \arg\min_T \sum_j \|x_j - \pi(T X_j)\|_2^2`$
Where $`\pi`$ is the projection function.

#### Loop Closure
**Similarity Score:**
$`S_{ij} = \text{similarity}(F_i, F_j)`$

> **Try it yourself!**
> Implement a simple triangulation or bundle adjustment on synthetic data. How does noise affect the reconstruction?

---

## 5. 3D Reconstruction

### Stereo Vision

#### Disparity Computation
**Disparity:**
$`d = x_l - x_r`$

**Depth:**
$`Z = \frac{f \cdot B}{d}`$
Where:
- $`f`$ is focal length
- $`B`$ is baseline
- $`d`$ is disparity

#### Stereo Matching
**Cost Function:**
$`C(x, y, d) = \|I_l(x, y) - I_r(x-d, y)\|`$

**Semi-Global Matching (SGM):**
$`L_r(p, d) = C(p, d) + \min(L_r(p-r, d), L_r(p-r, d\pm1) + P_1, \min_k L_r(p-r, k) + P_2)`$

### Multi-View Stereo (MVS)

#### PatchMatch Stereo
**Patch Similarity:**
$`S(p, q) = \sum_{i,j} w(i, j) \|I_1(p + (i,j)) - I_2(q + (i,j))\|_2`$

**Depth Refinement:**
$`d_{new} = d_{old} + \Delta d`$

#### COLMAP
**Photometric Consistency:**
$`C(p) = \sum_{i,j} \|I_i(p) - I_j(p)\|_2`$

**Geometric Consistency:**
$`G(p) = \sum_{i,j} \|D_i(p) - D_j(p)\|_2`$

### Deep Learning for 3D Reconstruction

#### Learning-based MVS
**Cost Volume:**
$`C(d) = \text{concat}(I_1, I_2(d), I_3(d), ... )`$

**Depth Prediction:**
$`D = \text{softmax}(\text{Conv3D}(C))`$

#### NeRF (Neural Radiance Fields)
**Volume Rendering:**
$`C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt`$

**Transmittance:**
$`T(t) = \exp\left(-\int_{t_n}^t \sigma(r(s)) ds\right)`$

> **Key Insight:**
> NeRF and learning-based MVS have revolutionized 3D reconstruction, enabling photorealistic novel view synthesis from sparse images.

---

## 6. Python Implementation Examples

Below are Python code examples for the main 3D vision techniques. Each function is annotated with comments to clarify the steps.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

# Create synthetic point cloud
def create_synthetic_point_cloud(num_points=1000):
    """Create synthetic 3D point cloud."""
    # Create a simple 3D shape (cube with noise)
    points = np.random.rand(num_points, 3) * 2 - 1  # Cube from -1 to 1
    
    # Add some structure
    # Create a sphere
    sphere_center = np.array([0.5, 0.5, 0.5])
    sphere_radius = 0.3
    sphere_points = np.random.randn(num_points//2, 3)
    sphere_points = sphere_points / np.linalg.norm(sphere_points, axis=1, keepdims=True) * sphere_radius
    sphere_points += sphere_center
    
    # Create a plane
    plane_points = np.random.rand(num_points//2, 3)
    plane_points[:, 2] = -0.5  # Fixed z-coordinate
    
    # Combine points
    all_points = np.vstack([sphere_points, plane_points])
    
    # Add noise
    noise = np.random.normal(0, 0.05, all_points.shape)
    all_points += noise
    
    return all_points

# Point cloud visualization
def visualize_point_cloud(points, title="Point Cloud"):
    """Visualize 3D point cloud."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.show()

# Point cloud downsampling
def downsample_point_cloud(points, target_size=500):
    """Downsample point cloud using random sampling."""
    if len(points) <= target_size:
        return points
    
    indices = np.random.choice(len(points), target_size, replace=False)
    return points[indices]

# Farthest Point Sampling (FPS)
def farthest_point_sampling(points, num_samples):
    """Implement Farthest Point Sampling."""
    if len(points) <= num_samples:
        return points
    
    # Initialize with random point
    sampled_indices = [np.random.randint(len(points))]
    remaining_indices = list(range(len(points)))
    remaining_indices.remove(sampled_indices[0])
    
    for _ in range(num_samples - 1):
        # Calculate distances to sampled points
        distances = cdist(points[remaining_indices], points[sampled_indices])
        min_distances = np.min(distances, axis=1)
        
        # Find farthest point
        farthest_idx = remaining_indices[np.argmax(min_distances)]
        sampled_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)
    
    return points[sampled_indices]

# Point cloud clustering
def cluster_point_cloud(points, eps=0.1, min_samples=5):
    """Cluster point cloud using DBSCAN."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    return labels

# Point cloud registration (ICP)
def iterative_closest_point(source, target, max_iterations=50, tolerance=1e-6):
    """Implement Iterative Closest Point algorithm."""
    # Initialize transformation
    transformation = np.eye(4)
    
    for iteration in range(max_iterations):
        # Find closest points
        distances = cdist(source, target)
        correspondences = np.argmin(distances, axis=1)
        
        # Calculate centroids
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target[correspondences], axis=0)
        
        # Center points
        source_centered = source - source_centroid
        target_centered = target[correspondences] - target_centroid
        
        # Calculate rotation matrix
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = target_centroid - R @ source_centroid
        
        # Update transformation
        current_transformation = np.eye(4)
        current_transformation[:3, :3] = R
        current_transformation[:3, 3] = t
        
        transformation = current_transformation @ transformation
        
        # Transform source points
        source_homogeneous = np.hstack([source, np.ones((len(source), 1))])
        source = (transformation @ source_homogeneous.T).T[:, :3]
        
        # Check convergence
        if np.linalg.norm(t) < tolerance:
            break
    
    return transformation, source

# Voxelization
def voxelize_point_cloud(points, voxel_size=0.1):
    """Convert point cloud to voxel grid."""
    # Calculate voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Find unique voxels
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    
    # Calculate voxel centers
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2
    
    return voxel_centers, unique_voxels

# Point cloud normal estimation
def estimate_normals(points, k_neighbors=10):
    """Estimate surface normals using PCA."""
    from sklearn.neighbors import NearestNeighbors
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    normals = np.zeros_like(points)
    
    for i in range(len(points)):
        # Get neighbors
        neighbor_points = points[indices[i]]
        
        # Calculate covariance matrix
        centered_points = neighbor_points - np.mean(neighbor_points, axis=0)
        cov_matrix = centered_points.T @ centered_points
        
        # Find eigenvector with smallest eigenvalue (normal direction)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue
        
        # Ensure consistent orientation (pointing outward)
        if normal[2] < 0:
            normal = -normal
        
        normals[i] = normal
    
    return normals

# Main demonstration
def demonstrate_point_cloud_processing():
    """Demonstrate various point cloud processing techniques."""
    # Create synthetic point cloud
    points = create_synthetic_point_cloud(2000)
    
    # Visualize original point cloud
    visualize_point_cloud(points, "Original Point Cloud")
    
    # Downsampling
    downsampled = downsample_point_cloud(points, target_size=500)
    visualize_point_cloud(downsampled, "Downsampled Point Cloud")
    
    # Farthest Point Sampling
    fps_points = farthest_point_sampling(points, num_samples=300)
    visualize_point_cloud(fps_points, "FPS Sampled Point Cloud")
    
    # Clustering
    labels = cluster_point_cloud(points, eps=0.15, min_samples=10)
    
    # Visualize clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:  # Noise points
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                      c='black', s=1, alpha=0.3)
        else:
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                      c=[color], s=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Clustered Point Cloud')
    plt.show()
    
    # Voxelization
    voxel_centers, voxel_indices = voxelize_point_cloud(points, voxel_size=0.1)
    visualize_point_cloud(voxel_centers, "Voxelized Point Cloud")
    
    # Normal estimation
    normals = estimate_normals(points, k_neighbors=15)
    
    # Visualize normals
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization
    sample_indices = np.random.choice(len(points), 100, replace=False)
    sample_points = points[sample_indices]
    sample_normals = normals[sample_indices]
    
    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
              c='blue', s=10)
    
    # Draw normals
    for point, normal in zip(sample_points, sample_normals):
        end_point = point + normal * 0.1
        ax.plot([point[0], end_point[0]], 
                [point[1], end_point[1]], 
                [point[2], end_point[2]], 'r-', linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud with Normals')
    plt.show()
    
    print(f"Original points: {len(points)}")
    print(f"Downsampled points: {len(downsampled)}")
    print(f"FPS points: {len(fps_points)}")
    print(f"Voxels: {len(voxel_centers)}")
    print(f"Clusters: {len(np.unique(labels)) - 1}")  # Exclude noise

# Stereo vision simulation
def simulate_stereo_vision():
    """Simulate stereo vision for 3D reconstruction."""
    # Create synthetic 3D points
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * np.sin(X) * np.cos(Y)  # Simple surface
    
    # Flatten and create point cloud
    points_3d = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Camera parameters
    focal_length = 1000
    baseline = 0.1
    image_width = 800
    image_height = 600
    
    # Project to left and right cameras
    left_points = []
    right_points = []
    
    for point in points_3d:
        # Left camera (at origin)
        if point[2] > 0:  # Only points in front of camera
            x_left = focal_length * point[0] / point[2] + image_width / 2
            y_left = focal_length * point[1] / point[2] + image_height / 2
            
            # Right camera (translated by baseline)
            x_right = focal_length * (point[0] - baseline) / point[2] + image_width / 2
            y_right = focal_length * point[1] / point[2] + image_height / 2
            
            if (0 <= x_left < image_width and 0 <= y_left < image_height and
                0 <= x_right < image_width and 0 <= y_right < image_height):
                left_points.append([x_left, y_left])
                right_points.append([x_right, y_right])
    
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    
    # Calculate disparity
    disparity = left_points[:, 0] - right_points[:, 0]
    
    # Reconstruct 3D points
    reconstructed_z = focal_length * baseline / disparity
    reconstructed_x = (left_points[:, 0] - image_width / 2) * reconstructed_z / focal_length
    reconstructed_y = (left_points[:, 1] - image_height / 2) * reconstructed_z / focal_length
    
    reconstructed_points = np.column_stack([reconstructed_x, reconstructed_y, reconstructed_z])
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original 3D surface
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap='viridis', s=1)
    ax1.set_title('Original 3D Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Left camera view
    axes[0, 1].scatter(left_points[:, 0], left_points[:, 1], c=disparity, cmap='viridis', s=1)
    axes[0, 1].set_title('Left Camera View')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].invert_yaxis()
    
    # Right camera view
    axes[1, 0].scatter(right_points[:, 0], right_points[:, 1], c=disparity, cmap='viridis', s=1)
    axes[1, 0].set_title('Right Camera View')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].invert_yaxis()
    
    # Reconstructed 3D surface
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], 
               c=reconstructed_points[:, 2], cmap='viridis', s=1)
    ax4.set_title('Reconstructed 3D Surface')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate reconstruction error
    error = np.mean(np.linalg.norm(points_3d[:len(reconstructed_points)] - reconstructed_points, axis=1))
    print(f"Average reconstruction error: {error:.4f}")

# Main execution
if __name__ == "__main__":
    # Demonstrate point cloud processing
    demonstrate_point_cloud_processing()
    
    # Demonstrate stereo vision
    simulate_stereo_vision()
```

> **Key Insight:**
> Understanding the code behind 3D vision helps demystify the algorithms and enables you to adapt them for your own projects.

---

## 7. Advanced 3D Vision Techniques

Advanced analysis includes multi-view stereo, robust registration, surface reconstruction, and 3D object detection.

- **Multi-View Stereo:** Reconstruct dense 3D geometry from multiple images.
- **RANSAC Registration:** Align point clouds robustly in the presence of outliers.
- **Surface Reconstruction:** Convert point clouds to meshes for visualization and simulation.
- **3D Object Detection:** Detect and localize objects in 3D space.

> **Try it yourself!**
> Use the provided code to experiment with stereo vision, RANSAC registration, and surface reconstruction. How do these methods handle noise and outliers?

---

## Summary Table

| Method         | Speed      | Accuracy   | Handles Noise | Real-Time? | Key Idea                |
|----------------|------------|------------|--------------|------------|-------------------------|
| PointNet       | Fast       | Medium     | No           | Yes        | Symmetric functions     |
| PointNet++     | Medium     | High       | Yes          | Yes        | Hierarchical features   |
| VoxelNet       | Medium     | High       | Yes          | No         | Voxelization            |
| PointPillars   | Very Fast  | High       | Yes          | Yes        | Pillar encoding         |
| SfM            | Slow       | High       | Yes          | No         | Multi-view geometry     |
| SLAM           | Medium     | High       | Yes          | Yes        | Mapping + localization  |
| Stereo Vision  | Fast       | Medium     | No           | Yes        | Disparity/depth         |
| NeRF           | Slow       | Very High  | Yes          | No         | Neural rendering        |

---

## Further Reading
- [Qi, C.R. et al. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [Qi, C.R. et al. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)
- [Zhou, Y. et al. (2018). VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396)
- [Lang, A.H. et al. (2019). PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
- [Mildenhall, B. et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934)

---

> **Next Steps:**
> - Experiment with different 3D vision methods on your own data.
> - Try combining point-based and voxel-based approaches for improved results.
> - Explore NeRF and learning-based MVS for photorealistic 3D reconstruction. 