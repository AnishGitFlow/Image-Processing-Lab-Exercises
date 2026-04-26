"""
Lab 1: Basic Image Processing Techniques using OpenCV and Matplotlib

Aim:
To apply and demonstrate basic image manipulation techniques including:
1. Resize an image to a specified size
2. Alter the aspect ratio of the image
3. Rotate the image by a specified angle
4. Flip the image both horizontally and vertically
5. Crop a region of interest (ROI) from the image
6. Convert the image to grayscale
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the user-provided image
img = cv2.imread('image1.jpg')

if img is None:
    print("Warning: 'image.jpg' not found. Using synthetic image for demonstration.")
    # Create a sample colored image for demonstration
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    # Add some colored regions
    img[50:150, 50:150] = [255, 0, 0]    # Blue region (BGR)
    img[50:150, 200:300] = [0, 255, 0]   # Green region
    img[180:280, 100:250] = [0, 0, 255]  # Red region
    cv2.putText(img, 'Sample Image', (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
else:
    print(f"Loaded image: image.jpg ({img.shape[1]}x{img.shape[0]})")

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 1. Resize the image to a specified size
resized_img = cv2.resize(img_rgb, (200, 150))

# 2. Alter the aspect ratio of the image (distorted resize)
aspect_ratio_changed = cv2.resize(img_rgb, (300, 150))  # Stretched horizontally

# 3. Rotate the image by a specified angle (45 degrees)
(h, w) = img_rgb.shape[:2]
center = (w // 2, h // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_img = cv2.warpAffine(img_rgb, rotation_matrix, (w, h))

# 4. Flip the image horizontally and vertically
flipped_horizontal = cv2.flip(img_rgb, 1)  # 1 = horizontal flip
flipped_vertical = cv2.flip(img_rgb, 0)    # 0 = vertical flip

# 5. Crop a region of interest (ROI) from the image
# Crop the center region (100x100 pixels)
start_y, end_y = 100, 200
start_x, end_x = 150, 250
cropped_img = img_rgb[start_y:end_y, start_x:end_x]

# 6. Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display all results
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Lab 1: Basic Image Processing Techniques', fontsize=16)

axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(resized_img)
axes[0, 1].set_title('1. Resized (200x150)')
axes[0, 1].axis('off')

axes[0, 2].imshow(aspect_ratio_changed)
axes[0, 2].set_title('2. Aspect Ratio Changed (300x150)')
axes[0, 2].axis('off')

axes[1, 0].imshow(rotated_img)
axes[1, 0].set_title('3. Rotated 45°')
axes[1, 0].axis('off')

axes[1, 1].imshow(flipped_horizontal)
axes[1, 1].set_title('4a. Flipped Horizontal')
axes[1, 1].axis('off')

axes[1, 2].imshow(flipped_vertical)
axes[1, 2].set_title('4b. Flipped Vertical')
axes[1, 2].axis('off')

axes[2, 0].imshow(cropped_img)
axes[2, 0].set_title('5. Cropped ROI (100x100)')
axes[2, 0].axis('off')

axes[2, 1].imshow(gray_img, cmap='gray')
axes[2, 1].set_title('6. Grayscale')
axes[2, 1].axis('off')

axes[2, 2].axis('off')

plt.tight_layout()
plt.savefig('lab1_output.png', dpi=150, bbox_inches='tight')
plt.show()

print("=" * 60)
print("LAB 1: BASIC IMAGE PROCESSING - COMPLETED")
print("=" * 60)
print("\nOperations performed:")
print("1. Resized image to (200, 150)")
print("2. Changed aspect ratio to (300, 150) - horizontally stretched")
print("3. Rotated image by 45 degrees around center")
print("4. Flipped image horizontally and vertically")
print("5. Cropped region of interest: [100:200, 150:250]")
print("6. Converted to grayscale")
print("\nOutput saved as: lab1_output.png")
print("=" * 60)
