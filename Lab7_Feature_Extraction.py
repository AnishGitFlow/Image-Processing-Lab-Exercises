"""
Lab 7: Feature Extraction Using GLCM, Color Histogram, LBP, and HOG

Aim:
To implement and analyze various feature extraction techniques:
1. GLCM (Gray Level Co-occurrence Matrix) - Texture features
2. Color Histogram - Color features
3. LBP (Local Binary Pattern) - Local texture patterns
4. HOG (Histogram of Oriented Gradients) - Shape/edge features
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from skimage import exposure

# Load the user-provided image
user_img = cv2.imread('image7.jpg')

if user_img is None:
    print("Warning: 'image.jpg' not found. Using synthetic images for demonstration.")
    np.random.seed(42)
    # Create sample images for different feature extraction methods
    
    # 1. Texture image for GLCM and LBP
    texture_img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            val = min(((i + j) // 32) * 30 + 30, 255)
            texture_img[i:i+32, j:j+32] = val
    noise = np.random.normal(0, 10, texture_img.shape).astype(np.int16)
    texture_img = np.clip(texture_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 2. Color image for Color Histogram
    color_img = np.zeros((256, 256, 3), dtype=np.uint8)
    color_img[:85, :, 0] = 200
    color_img[85:170, :, 1] = 180
    color_img[170:, :, 2] = 220
    cv2.circle(color_img, (128, 128), 50, (255, 255, 0), -1)
    
    # 3. Shape image for HOG
    shape_img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(shape_img, (50, 50), (150, 150), 200, 2)
    cv2.circle(shape_img, (180, 180), 50, 150, 2)
    cv2.line(shape_img, (30, 200), (220, 100), 255, 2)
else:
    print(f"Loaded image: image.jpg ({user_img.shape[1]}x{user_img.shape[0]})")
    # Use user image for all feature extraction (resize to consistent sizes)
    texture_img = cv2.cvtColor(user_img, cv2.COLOR_BGR2GRAY)
    texture_img = cv2.resize(texture_img, (256, 256))
    color_img = cv2.resize(user_img, (256, 256))
    shape_img = cv2.resize(texture_img, (256, 256))

# Convert color image to RGB for display
color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

# ==================== 1. GLCM Features ====================
# Calculate GLCM
distances = [1]
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm = graycomatrix(texture_img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

# Extract properties
contrast = graycoprops(glcm, 'contrast')[0, 0]
dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]

# ==================== 2. Color Histogram ====================
color_hist_r = cv2.calcHist([color_img_rgb], [0], None, [256], [0, 256])
color_hist_g = cv2.calcHist([color_img_rgb], [1], None, [256], [0, 256])
color_hist_b = cv2.calcHist([color_img_rgb], [2], None, [256], [0, 256])

# ==================== 3. LBP Features ====================
# Parameters for LBP
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(texture_img, n_points, radius, method='uniform')

# Calculate LBP histogram
lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
lbp_hist = lbp_hist.astype("float")
lbp_hist /= (lbp_hist.sum() + 1e-7)

# ==================== 4. HOG Features ====================
fd, hog_image = hog(shape_img, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=True)

# Rescale HOG image for display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# ==================== Visualization ====================
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# Row 1: GLCM
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(texture_img, cmap='gray')
ax1.set_title('1. Texture Image (for GLCM)')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
# Display a slice of GLCM for visualization
glcm_vis = glcm[:, :, 0, 0]  # First distance and angle
ax2.imshow(np.log1p(glcm_vis), cmap='hot')
ax2.set_title('GLCM (d=1, θ=0°)')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
glcm_text = f"GLCM Features:\n" \
            f"Contrast: {contrast:.3f}\n" \
            f"Dissimilarity: {dissimilarity:.3f}\n" \
            f"Homogeneity: {homogeneity:.3f}\n" \
            f"Energy: {energy:.3f}\n" \
            f"Correlation: {correlation:.3f}"
ax3.text(0.1, 0.5, glcm_text, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Row 2: Color Histogram
ax4 = fig.add_subplot(gs[1, 0])
ax4.imshow(color_img_rgb)
ax4.set_title('2. Color Image')
ax4.axis('off')

ax5 = fig.add_subplot(gs[1, 1:])
ax5.plot(color_hist_r, color='r', label='Red', alpha=0.7)
ax5.plot(color_hist_g, color='g', label='Green', alpha=0.7)
ax5.plot(color_hist_b, color='b', label='Blue', alpha=0.7)
ax5.set_xlim([0, 256])
ax5.set_title('Color Histogram (RGB)')
ax5.set_xlabel('Pixel Intensity')
ax5.set_ylabel('Frequency')
ax5.legend()

# Row 3: LBP
ax6 = fig.add_subplot(gs[2, 0])
ax6.imshow(texture_img, cmap='gray')
ax6.set_title('3. Original (for LBP)')
ax6.axis('off')

ax7 = fig.add_subplot(gs[2, 1])
ax7.imshow(lbp, cmap='nipy_spectral')
ax7.set_title(f'LBP Image (r={radius}, p={n_points})')
ax7.axis('off')

ax8 = fig.add_subplot(gs[2, 2])
ax8.bar(range(len(lbp_hist)), lbp_hist, color='steelblue')
ax8.set_title('LBP Histogram')
ax8.set_xlabel('LBP Pattern')
ax8.set_ylabel('Normalized Frequency')

# Row 4: HOG
ax9 = fig.add_subplot(gs[3, 0])
ax9.imshow(shape_img, cmap='gray')
ax9.set_title('4. Shape Image (for HOG)')
ax9.axis('off')

ax10 = fig.add_subplot(gs[3, 1])
ax10.imshow(hog_image_rescaled, cmap='gray')
ax10.set_title(f'HOG Visualization\n({len(fd)} features)')
ax10.axis('off')

ax11 = fig.add_subplot(gs[3, 2])
ax11.axis('off')
hog_text = f"HOG Features:\n" \
           f"Orientations: 9\n" \
           f"Pixels per cell: 16×16\n" \
           f"Cells per block: 2×2\n" \
           f"Total features: {len(fd)}"
ax11.text(0.1, 0.5, hog_text, fontsize=11, verticalalignment='center',
          bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

fig.suptitle('Lab 7: Feature Extraction Techniques', fontsize=16, y=0.98)
plt.savefig('lab7_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("LAB 7: FEATURE EXTRACTION TECHNIQUES")
print("=" * 60)

print("\n1. GLCM (Gray Level Co-occurrence Matrix)")
print("   Statistical texture features:")
print(f"   - Contrast: {contrast:.3f} (measures local intensity variation)")
print(f"   - Dissimilarity: {dissimilarity:.3f} (measures local gray level difference)")
print(f"   - Homogeneity: {homogeneity:.3f} (measures local gray level uniformity)")
print(f"   - Energy: {energy:.3f} (measures uniformity of texture)")
print(f"   - Correlation: {correlation:.3f} (measures linear dependency)")

print("\n2. Color Histogram")
print("   Color distribution features:")
print(f"   - Red channel mean: {np.mean(color_hist_r):.1f}")
print(f"   - Green channel mean: {np.mean(color_hist_g):.1f}")
print(f"   - Blue channel mean: {np.mean(color_hist_b):.1f}")
print(f"   - Total bins: 256 per channel")

print("\n3. LBP (Local Binary Pattern)")
print("   Local texture descriptor:")
print(f"   - Radius: {radius}")
print(f"   - Sampling points: {n_points}")
print(f"   - Uniform patterns: {len(lbp_hist)}")
print(f"   - Most frequent pattern: {np.argmax(lbp_hist)}")

print("\n4. HOG (Histogram of Oriented Gradients)")
print("   Shape and edge descriptor:")
print(f"   - Orientations: 9")
print(f"   - Pixels per cell: 16×16")
print(f"   - Cells per block: 2×2")
print(f"   - Total HOG features: {len(fd)}")

print("\nOutput saved as: lab7_output.png")
print("=" * 60)
