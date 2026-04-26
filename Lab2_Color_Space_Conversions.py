"""
Lab 2: Color Space Conversions in Image Processing using OpenCV

Aim:
To explore and demonstrate various color space conversions:
1. RGB to HSV, HSL, YCrCb, Lab, and XYZ
2. RGB to Grayscale
3. Grayscale to other color spaces (RGB, HSV, Lab)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the user-provided image
img = cv2.imread('image2.jpg')

if img is None:
    print("Warning: 'image.jpg' not found. Using synthetic image for demonstration.")
    # Create a sample image with various colors for demonstration
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[50:150, 50:150] = [255, 0, 0]      # Blue (BGR)
    img[50:150, 200:300] = [0, 255, 0]     # Green
    img[180:280, 50:150] = [0, 0, 255]     # Red
    img[180:280, 200:300] = [255, 255, 0]  # Yellow
    img[120:200, 150:250] = [255, 0, 255] # Magenta
    cv2.putText(img, 'Color Spaces', (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
else:
    print(f"Loaded image: image.jpg ({img.shape[1]}x{img.shape[0]})")

# Convert BGR to RGB for matplotlib display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Color Space Conversions
# 1. RGB to HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 2. RGB to HSL (Note: OpenCV uses HLS format)
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# 3. RGB to YCrCb
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# 4. RGB to Lab
img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# 5. RGB to XYZ
img_xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

# 6. RGB to Grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 7. Grayscale back to RGB (3-channel)
gray_to_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

# 8. Grayscale to HSV (via RGB intermediate)
gray_to_hsv = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
gray_to_hsv = cv2.cvtColor(gray_to_hsv, cv2.COLOR_BGR2HSV)

# 9. Grayscale to Lab (via RGB intermediate)
gray_to_lab = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
gray_to_lab = cv2.cvtColor(gray_to_lab, cv2.COLOR_BGR2LAB)

# Display results
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Lab 2: Color Space Conversions', fontsize=16)

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original RGB')
axes[0, 0].axis('off')

axes[0, 1].imshow(img_hsv)
axes[0, 1].set_title('1. RGB to HSV')
axes[0, 1].axis('off')

axes[0, 2].imshow(img_hls)
axes[0, 2].set_title('2. RGB to HSL (HLS)')
axes[0, 2].axis('off')

axes[0, 3].imshow(img_ycrcb)
axes[0, 3].set_title('3. RGB to YCrCb')
axes[0, 3].axis('off')

axes[1, 0].imshow(img_lab)
axes[1, 0].set_title('4. RGB to Lab')
axes[1, 0].axis('off')

axes[1, 1].imshow(img_xyz)
axes[1, 1].set_title('5. RGB to XYZ')
axes[1, 1].axis('off')

axes[1, 2].imshow(img_gray, cmap='gray')
axes[1, 2].set_title('6. RGB to Grayscale')
axes[1, 2].axis('off')

axes[1, 3].imshow(gray_to_rgb)
axes[1, 3].set_title('7. Grayscale to RGB')
axes[1, 3].axis('off')

axes[2, 0].imshow(gray_to_hsv)
axes[2, 0].set_title('8. Grayscale to HSV')
axes[2, 0].axis('off')

axes[2, 1].imshow(gray_to_lab)
axes[2, 1].set_title('9. Grayscale to Lab')
axes[2, 1].axis('off')

axes[2, 2].axis('off')
axes[2, 3].axis('off')

plt.tight_layout()
plt.savefig('lab2_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("LAB 2: COLOR SPACE CONVERSIONS - COMPLETED")
print("=" * 60)
print("\nColor Space Conversions performed:")
print("1. RGB to HSV - Hue, Saturation, Value")
print("2. RGB to HSL (HLS in OpenCV) - Hue, Saturation, Lightness")
print("3. RGB to YCrCb - Used in video compression")
print("4. RGB to Lab - Device-independent, perceptually uniform")
print("5. RGB to XYZ - Based on human visual perception")
print("6. RGB to Grayscale - Single channel luminance")
print("7. Grayscale to RGB - 3-channel conversion")
print("8. Grayscale to HSV - via RGB intermediate")
print("9. Grayscale to Lab - via RGB intermediate")
print("\nOutput saved as: lab2_output.png")
print("=" * 60)
