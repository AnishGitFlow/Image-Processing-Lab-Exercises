# Image Processing Lab Exercises

A comprehensive collection of computer vision and image processing laboratory exercises implemented in Python using OpenCV, NumPy, Matplotlib, scikit-image, and scikit-learn.

## Overview

This repository contains 7 hands-on labs covering fundamental to advanced image processing techniques, from basic image manipulation to feature extraction algorithms.

## Requirements

```bash
pip install opencv-python numpy matplotlib scikit-image scikit-learn
```

**Dependencies:**
- Python 3.7+
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-image
- scikit-learn

## Lab Contents

### Lab 1: Basic Image Processing Techniques
**File:** `Lab1_Basic_Image_Processing.py`

Demonstrates fundamental image manipulation operations:
- Image resizing and aspect ratio modification
- Rotation by specified angles
- Horizontal and vertical flipping
- Region of Interest (ROI) cropping
- Grayscale conversion

**Usage:** Place an image named `image1.jpg` in the same directory, or the script will generate a synthetic image.

---

### Lab 2: Color Space Conversions
**File:** `Lab2_Color_Space_Conversions.py`

Explores various color space transformations:
- RGB to HSV (Hue, Saturation, Value)
- RGB to HSL/HLS (Hue, Saturation, Lightness)
- RGB to YCrCb (Video compression standard)
- RGB to Lab (Perceptually uniform, device-independent)
- RGB to XYZ (CIE standard, human visual perception)
- RGB to Grayscale conversion
- Grayscale to other color spaces

**Usage:** Place an image named `image2.jpg` in the same directory, or the script will generate a synthetic color test image.

---

### Lab 3: Histogram Equalization & Contrast Enhancement
**File:** `Lab3_Histogram_Equalization.py`

Implements image enhancement techniques for improving contrast:
- Histogram Equalization (global contrast enhancement)
- Contrast Stretching (linear normalization)
- CLAHE - Contrast Limited Adaptive Histogram Equalization

**Usage:** Place an image named `image3.jpg` in the same directory, or the script will generate a synthetic low-contrast image.

---

### Lab 4: Image Enhancement Transformations
**File:** `Lab4_Image_Enhancement_Transforms.py`

Demonstrates intensity transformation techniques:
- **Negative Transformation**: Inverts pixel values (s = 255 - r)
- **Logarithmic Transformation**: Expands dark region details (s = c * log(1 + r))
- **Gamma Correction**: Non-linear brightness adjustment with variable gamma values
  - γ < 1: Brightens dark regions
  - γ > 1: Darkens bright regions
- **Contrast Stretching**: Linear normalization to full dynamic range

**Usage:** Place an image named `image4.jpg` in the same directory, or the script will generate a synthetic image with varying intensities.

---

### Lab 5: K-Means Clustering for Image Segmentation
**File:** `Lab5_KMeans_Segmentation.py`

Implements color-based image segmentation using machine learning:
- K-Means clustering algorithm for pixel grouping
- Segmentation with varying K values (2, 4, 6, 8 clusters)
- Dominant color extraction and palette visualization
- Color quantization and region-based segmentation

**Dependencies:** scikit-learn (KMeans)

**Usage:** Place an image named `image5.jpg` in the same directory, or the script will generate a synthetic multi-colored image.

---

### Lab 6: Edge Detection Techniques
**File:** `Lab6_Edge_Detection.py`

Compares different edge detection algorithms:
- **Canny Edge Detection**: Multi-stage algorithm with Gaussian blur, gradient calculation, non-maximum suppression, and hysteresis thresholding
- **Sobel Edge Detection**: First derivative operator for detecting horizontal and vertical edges
- **Laplacian Edge Detection**: Second derivative operator detecting rapid intensity changes

**Usage:** Place an image named `image6.jpg` in the same directory, or the script will generate a synthetic image with geometric shapes.

---

### Lab 7: Feature Extraction Techniques
**File:** `Lab7_Feature_Extraction.py`

Implements multiple feature extraction methods for image analysis:
- **GLCM (Gray Level Co-occurrence Matrix)**: Statistical texture features (contrast, dissimilarity, homogeneity, energy, correlation)
- **Color Histogram**: RGB color distribution analysis
- **LBP (Local Binary Pattern)**: Local texture descriptor for pattern recognition
- **HOG (Histogram of Oriented Gradients)**: Shape and edge feature descriptor

**Dependencies:** scikit-image (graycomatrix, graycoprops, local_binary_pattern, hog)

**Usage:** Place an image named `image7.jpg` in the same directory, or the script will generate synthetic test images.

---

## How to Run

1. Clone or download this repository
2. Install the required dependencies
3. (Optional) Add your own images named `image1.jpg` through `image7.jpg` for each respective lab
4. Run any lab script:

```bash
python Lab1_Basic_Image_Processing.py
```

Each script will:
- Load your provided image (if available) or generate a synthetic demonstration image
- Process the image using the specific techniques
- Display results using Matplotlib
- Save output figures as PNG files (e.g., `lab1_output.png`)
- Print detailed statistics to the console

## Output Files

Each lab generates visualization outputs:
- `lab1_output.png` - Basic processing results
- `lab2_output.png` - Color space conversions
- `lab3_output.png` - Histogram equalization comparison
- `lab4_output.png` - Enhancement transformations
- `lab5_output.png`, `lab5_color_palette.png` - Segmentation results
- `lab6_output.png`, `lab6_comparison.png` - Edge detection comparison
- `lab7_output.png` - Feature extraction visualization

## Learning Outcomes

After completing these labs, you will understand:
- Core image manipulation operations with OpenCV
- Color theory and color space transformations
- Contrast enhancement and histogram techniques
- Intensity transformations for image improvement
- Machine learning applications in image segmentation
- Edge detection theory and practical implementation
- Feature extraction for computer vision tasks

## License

This project is intended for educational purposes.

## Author

Image Processing Lab Exercises - Educational Collection
