"""
Lab 3: Image Enhancement using Histogram Equalization and Contrast Stretching

Aim:
To explore image enhancement techniques on grayscale images:
1. Convert image to grayscale
2. Apply histogram equalization
3. Perform contrast stretching
4. Apply Adaptive Histogram Equalization (CLAHE)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the user-provided image
img_color = cv2.imread('image3.jpg')

if img_color is None:
    print("Warning: 'image.jpg' not found. Using synthetic image for demonstration.")
    # Create a sample low-contrast grayscale image
    np.random.seed(42)
    img = np.zeros((300, 400), dtype=np.uint8)
    # Create regions with different intensities (low contrast)
    img[50:150, 50:200] = 80
    img[50:150, 220:350] = 120
    img[180:280, 50:200] = 140
    img[180:280, 220:350] = 100
    # Add some gradient for better visualization
    for i in range(300):
        img[i, :] = np.clip(img[i, :].astype(np.int16) + int(20 * np.sin(i / 30)), 0, 255).astype(np.uint8)
    # Add some noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
else:
    print(f"Loaded image: image.jpg ({img_color.shape[1]}x{img_color.shape[0]})")
    # Convert to grayscale
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 1. Histogram Equalization
equalized = cv2.equalizeHist(img)

# 2. Contrast Stretching (manual implementation)
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

stretched = contrast_stretching(img)

# 3. Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img)

# Create figure with images and histograms
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Original
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(img, cmap='gray')
ax1.set_title('Original Image (Low Contrast)')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1:])
ax2.hist(img.ravel(), 256, [0, 256], color='gray', alpha=0.7)
ax2.set_title('Original Histogram')
ax2.set_xlim([0, 256])

# Row 2: Histogram Equalization
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(equalized, cmap='gray')
ax3.set_title('Histogram Equalization')
ax3.axis('off')

ax4 = fig.add_subplot(gs[1, 1:])
ax4.hist(equalized.ravel(), 256, [0, 256], color='blue', alpha=0.7)
ax4.set_title('Equalized Histogram')
ax4.set_xlim([0, 256])

# Row 3: Contrast Stretching
ax5 = fig.add_subplot(gs[2, 0])
ax5.imshow(stretched, cmap='gray')
ax5.set_title('Contrast Stretching')
ax5.axis('off')

ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(stretched.ravel(), 256, [0, 256], color='green', alpha=0.7)
ax6.set_title('Stretched Histogram')
ax6.set_xlim([0, 256])

# Row 3: CLAHE
ax7 = fig.add_subplot(gs[2, 2])
ax7.imshow(clahe_img, cmap='gray')
ax7.set_title('CLAHE (Adaptive)')
ax7.axis('off')

fig.suptitle('Lab 3: Histogram Equalization & Contrast Enhancement', fontsize=16)
plt.savefig('lab3_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("LAB 3: HISTOGRAM EQUALIZATION & CONTRAST ENHANCEMENT")
print("=" * 60)
print("\nTechniques applied:")
print("1. Original Image - Low contrast synthetic image")
print(f"   Min intensity: {np.min(img)}, Max intensity: {np.max(img)}")
print("\n2. Histogram Equalization - Global contrast enhancement")
print(f"   Min intensity: {np.min(equalized)}, Max intensity: {np.max(equalized)}")
print("\n3. Contrast Stretching - Linear normalization")
print(f"   Min intensity: {np.min(stretched)}, Max intensity: {np.max(stretched)}")
print("\n4. CLAHE - Local adaptive histogram equalization")
print(f"   Clip Limit: 2.0, Tile Grid: 8x8")
print(f"   Min intensity: {np.min(clahe_img)}, Max intensity: {np.max(clahe_img)}")
print("\nOutput saved as: lab3_output.png")
print("=" * 60)
