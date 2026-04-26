"""
Lab 6: Edge Detection Techniques in Image Processing

Aim:
To compare various edge detection techniques:
1. Canny Edge Detection
2. Sobel Edge Detection (X and Y directions)
3. Laplacian Edge Detection
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the user-provided image
img_color = cv2.imread('image6.jpg')

if img_color is None:
    print("Warning: 'image.jpg' not found. Using synthetic image for demonstration.")
    # Create a sample image with shapes for edge detection
    img = np.zeros((300, 400), dtype=np.uint8)
    # Draw geometric shapes
    cv2.rectangle(img, (50, 50), (150, 150), 200, -1)  # Filled rectangle
    cv2.circle(img, (280, 100), 60, 150, -1)  # Filled circle
    cv2.line(img, (50, 250), (350, 200), 255, 3)  # Line
    cv2.ellipse(img, (200, 220), (80, 40), 30, 0, 360, 180, 2)
    # Add gradient region
    for i in range(100):
        img[50+i, 300:380] = int(50 + i * 1.5)
else:
    print(f"Loaded image: image.jpg ({img_color.shape[1]}x{img_color.shape[0]})")
    # Convert to grayscale
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

# 1. Canny Edge Detection
canny_edges = cv2.Canny(img_blur, 50, 150)

# 2. Sobel Edge Detection
sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

# Convert to absolute values and scale
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# Combine Sobel X and Y
sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

# 3. Laplacian Edge Detection
laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
laplacian_abs = cv2.convertScaleAbs(laplacian)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Lab 6: Edge Detection Techniques', fontsize=16)

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(img_blur, cmap='gray')
axes[0, 1].set_title('Blurred Image\n(Gaussian)')
axes[0, 1].axis('off')

axes[0, 2].imshow(canny_edges, cmap='gray')
axes[0, 2].set_title('Canny Edge Detection\n(th1=50, th2=150)')
axes[0, 2].axis('off')

axes[1, 0].imshow(sobel_x_abs, cmap='gray')
axes[1, 0].set_title('Sobel X (Horizontal Edges)')
axes[1, 0].axis('off')

axes[1, 1].imshow(sobel_y_abs, cmap='gray')
axes[1, 1].set_title('Sobel Y (Vertical Edges)')
axes[1, 1].axis('off')

axes[1, 2].imshow(laplacian_abs, cmap='gray')
axes[1, 2].set_title('Laplacian\n(2nd Derivative)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('lab6_output.png', dpi=150, bbox_inches='tight')
plt.show()

# Additional comparison figure
fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))

axes2[0].imshow(img, cmap='gray')
axes2[0].set_title('Original')
axes2[0].axis('off')

axes2[1].imshow(canny_edges, cmap='gray')
axes2[1].set_title('Canny\n(Multi-stage)')
axes2[1].axis('off')

axes2[2].imshow(sobel_combined, cmap='gray')
axes2[2].set_title('Sobel Combined\n(Gradient-based)')
axes2[2].axis('off')

axes2[3].imshow(laplacian_abs, cmap='gray')
axes2[3].set_title('Laplacian\n(2nd order)')
axes2[3].axis('off')

fig2.suptitle('Edge Detection Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('lab6_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("LAB 6: EDGE DETECTION TECHNIQUES")
print("=" * 60)
print("\nMethods compared:")
print("\n1. Canny Edge Detection")
print("   Steps: Gaussian blur → Gradient calculation →")
print("          Non-maximum suppression → Hysteresis thresholding")
print("   Thresholds: 50 (low), 150 (high)")
print(f"   Edges detected: {np.sum(canny_edges > 0)} pixels")

print("\n2. Sobel Edge Detection")
print("   First derivative operator using 3x3 kernels")
print("   Sobel X: Detects vertical changes (horizontal edges)")
print("   Sobel Y: Detects horizontal changes (vertical edges)")
print(f"   Mean gradient magnitude: X={np.mean(sobel_x_abs):.2f}, Y={np.mean(sobel_y_abs):.2f}")

print("\n3. Laplacian Edge Detection")
print("   Second derivative operator (detects rapid intensity changes)")
print(f"   Mean magnitude: {np.mean(laplacian_abs):.2f}")
print("   Note: More sensitive to noise")

print("\nOutput saved as: lab6_output.png, lab6_comparison.png")
print("=" * 60)
