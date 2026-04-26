"""
Lab 4: Image Enhancement Techniques
Negative, Logarithmic, Gamma, and Contrast Stretching Transformations

Aim:
To demonstrate various image enhancement techniques on grayscale images:
1. Negative Transformation
2. Logarithmic Transformation
3. Gamma Transformation
4. Contrast Stretching
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the user-provided image
img_color = cv2.imread('image4.jpg')

if img_color is None:
    print("Warning: 'image.jpg' not found. Using synthetic image for demonstration.")
    # Create a sample image with varying intensities
    np.random.seed(42)
    img = np.zeros((300, 400), dtype=np.uint8)
    # Create dark regions (low intensity) and bright regions
    img[50:150, 50:200] = 30   # Dark region
    img[50:150, 220:350] = 80  # Medium-dark
    img[180:280, 50:200] = 150 # Medium
    img[180:280, 220:350] = 220 # Bright
    # Add subtle gradients
    for i in range(300):
        img[i, :] = np.clip(img[i, :].astype(np.int16) + int(10 * np.sin(i / 20)), 0, 255).astype(np.uint8)
else:
    print(f"Loaded image: image.jpg ({img_color.shape[1]}x{img_color.shape[0]})")
    # Convert to grayscale
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 1. Negative Transformation
# Formula: s = L - 1 - r, where L = 256
negative = 255 - img

# 2. Logarithmic Transformation
# Formula: s = c * log(1 + r), where c = 255 / log(1 + max(r))
c_log = 255 / np.log(1 + np.max(img))
log_transformed = c_log * np.log(1 + img.astype(np.float32))
log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)

# 3. Gamma Transformation with different gamma values
# Formula: s = c * r^gamma
def gamma_correction(image, gamma):
    c = 255 / (255 ** gamma)
    gamma_corrected = c * (image.astype(np.float32) ** gamma)
    return np.clip(gamma_corrected, 0, 255).astype(np.uint8)

gamma_0_5 = gamma_correction(img, 0.5)  # Gamma < 1: brighten dark regions
gamma_2_0 = gamma_correction(img, 2.0)  # Gamma > 1: darken bright regions

# 4. Contrast Stretching (Piecewise Linear)
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return image
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

stretched = contrast_stretching(img)

# Display results
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
fig.suptitle('Lab 4: Image Enhancement Transformations', fontsize=16)

axes[0, 0].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(negative, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title('1. Negative Transform\ns = 255 - r')
axes[0, 1].axis('off')

axes[0, 2].imshow(log_transformed, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title('2. Logarithmic Transform\ns = c*log(1+r)')
axes[0, 2].axis('off')

axes[0, 3].imshow(stretched, cmap='gray', vmin=0, vmax=255)
axes[0, 3].set_title('3. Contrast Stretching\nLinear normalization')
axes[0, 3].axis('off')

axes[1, 0].imshow(gamma_0_5, cmap='gray', vmin=0, vmax=255)
axes[1, 0].set_title(r'4a. Gamma Correction ($\gamma$=0.5)')
axes[1, 0].axis('off')

axes[1, 1].imshow(img, cmap='gray', vmin=0, vmax=255)
axes[1, 1].set_title('Original (Reference)')
axes[1, 1].axis('off')

axes[1, 2].imshow(gamma_2_0, cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title(r'4b. Gamma Correction ($\gamma$=2.0)')
axes[1, 2].axis('off')

axes[1, 3].axis('off')

# Add intensity range information
def add_stats(ax, image, row, col):
    stats_text = f'Min: {np.min(image)}\nMax: {np.max(image)}\nMean: {np.mean(image):.1f}'
    ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, ha='center', fontsize=9)

add_stats(axes[0, 0], img, 0, 0)
add_stats(axes[0, 1], negative, 0, 1)
add_stats(axes[0, 2], log_transformed, 0, 2)
add_stats(axes[0, 3], stretched, 0, 3)
add_stats(axes[1, 0], gamma_0_5, 1, 0)
add_stats(axes[1, 1], img, 1, 1)
add_stats(axes[1, 2], gamma_2_0, 1, 2)

plt.tight_layout()
plt.savefig('lab4_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("LAB 4: IMAGE ENHANCEMENT TRANSFORMATIONS")
print("=" * 60)
print("\nTransformations applied:")
print("\n1. Negative Transformation")
print("   Formula: s = L - 1 - r (where L=256)")
print(f"   Input range: [{np.min(img)}, {np.max(img)}]")
print(f"   Output range: [{np.min(negative)}, {np.max(negative)}]")

print("\n2. Logarithmic Transformation")
print("   Formula: s = c * log(1 + r)")
print(f"   c = {c_log:.2f}")
print(f"   Output range: [{np.min(log_transformed)}, {np.max(log_transformed)}]")

print("\n3. Contrast Stretching")
print("   Linear normalization to full range [0, 255]")
print(f"   Output range: [{np.min(stretched)}, {np.max(stretched)}]")

print("\n4. Gamma Transformation")
print("   Formula: s = c * r^gamma")
print(f"   Gamma = 0.5: Brightens dark areas, range [{np.min(gamma_0_5)}, {np.max(gamma_0_5)}]")
print(f"   Gamma = 2.0: Darkens bright areas, range [{np.min(gamma_2_0)}, {np.max(gamma_2_0)}]")

print("\nOutput saved as: lab4_output.png")
print("=" * 60)
