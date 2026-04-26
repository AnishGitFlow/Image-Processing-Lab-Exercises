"""
Lab 5: K-Means Clustering for Image Segmentation

Aim:
To implement K-Means clustering for image segmentation by grouping similar colors.
The image is segmented into K distinct regions based on dominant color clusters.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the user-provided image
img = cv2.imread('image5.jpg')

if img is None:
    print("Warning: 'image.jpg' not found. Using synthetic image for demonstration.")
    # Create a sample image with distinct color regions
    np.random.seed(42)
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Define color regions (BGR format)
    colors = {
        'Blue': [255, 0, 0],
        'Green': [0, 255, 0],
        'Red': [0, 0, 255],
        'Yellow': [0, 255, 255],
        'Cyan': [255, 255, 0],
        'Magenta': [255, 0, 255]
    }
    # Create colored regions
    regions = [
        (0, 100, 0, 133, colors['Blue']),
        (0, 100, 134, 266, colors['Green']),
        (0, 100, 267, 400, colors['Red']),
        (101, 200, 0, 133, colors['Yellow']),
        (101, 200, 134, 266, colors['Cyan']),
        (101, 200, 267, 400, colors['Magenta']),
        (201, 300, 0, 200, [128, 128, 128]),  # Gray
        (201, 300, 201, 400, [255, 255, 255]), # White
    ]
    for y1, y2, x1, x2, color in regions:
        img[y1:y2, x1:x2] = color
        # Add some noise for realism
        noise = np.random.normal(0, 15, (y2-y1, x2-x1, 3)).astype(np.int16)
        img[y1:y2, x1:x2] = np.clip(img[y1:y2, x1:x2].astype(np.int16) + noise, 0, 255).astype(np.uint8)
else:
    print(f"Loaded image: image.jpg ({img.shape[1]}x{img.shape[0]})")

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape image for K-Means
pixels = img_rgb.reshape((-1, 3))
pixels = np.float32(pixels)

# Apply K-Means with different K values
K_values = [2, 4, 6, 8]
segmented_images = []

for K in K_values:
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Get cluster centers (dominant colors)
    centers = np.uint8(kmeans.cluster_centers_)
    
    # Create segmented image
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(img_rgb.shape)
    segmented_images.append(segmented)
    
    print(f"K={K}: Dominant colors (RGB): {centers.tolist()}")

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Lab 5: K-Means Clustering for Image Segmentation', fontsize=16)

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

for idx, (K, seg_img) in enumerate(zip(K_values, segmented_images)):
    row = (idx + 1) // 3
    col = (idx + 1) % 3
    axes[row, col].imshow(seg_img)
    axes[row, col].set_title(f'Segmented (K={K})')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('lab5_output.png', dpi=150, bbox_inches='tight')
plt.show()

# Additional visualization: Show color palette for K=6
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

axes2[0].imshow(img_rgb)
axes2[0].set_title('Original Image')
axes2[0].axis('off')

axes2[1].imshow(segmented_images[2])  # K=6 result
axes2[1].set_title('Segmented Image (K=6)')
axes2[1].axis('off')

# Get colors for K=6
kmeans_6 = KMeans(n_clusters=6, random_state=42, n_init=10)
labels_6 = kmeans_6.fit_predict(pixels)
centers_6 = np.uint8(kmeans_6.cluster_centers_)

# Show color palette
fig3, ax3 = plt.subplots(figsize=(8, 2))
ax3.set_title('Dominant Colors (K=6 Palette)')
for i, color in enumerate(centers_6):
    ax3.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color/255))
    ax3.text(i + 0.5, 0.5, f'{i+1}', ha='center', va='center', fontsize=12, color='white' if np.mean(color) < 128 else 'black')
ax3.set_xlim(0, 6)
ax3.set_ylim(0, 1)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_aspect('equal')
plt.tight_layout()
plt.savefig('lab5_color_palette.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("LAB 5: K-MEANS CLUSTERING FOR IMAGE SEGMENTATION")
print("=" * 60)
print("\nK-Means Algorithm Steps:")
print("1. Select K random centroids in RGB color space")
print("2. Assign each pixel to nearest centroid")
print("3. Update centroids as mean of assigned pixels")
print("4. Repeat until convergence")
print("\nResults with different K values:")
for i, K in enumerate(K_values):
    print(f"   K={K}: Image segmented into {K} color regions")
print("\nOutput saved as: lab5_output.png, lab5_color_palette.png")
print("=" * 60)
