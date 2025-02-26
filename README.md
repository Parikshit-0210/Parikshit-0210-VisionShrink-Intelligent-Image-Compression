# VisionShrink: Intelligent Image Compression

## Overview
### Image Compression Using SVD, PCA, and NMF

This project implements image compression techniques using three different methods:

- **Singular Value Decomposition (SVD)**
- **Principal Component Analysis (PCA)**
- **Non-negative Matrix Factorization (NMF)**

Additionally, the project includes functionalities for displaying image information, resizing images, and cropping images.

## Features

- **Image Compression**:
  - SVD-based compression
  - PCA-based compression
  - NMF-based compression
- **Image Processing**:
  - Display image information (format, size, mode)
  - Resize images
  - Crop images

## Installation

### Prerequisites

Ensure you have Python installed along with the required dependencies.

### Required Libraries

Install the necessary libraries using the following command:

```bash
pip install numpy pillow matplotlib
```

## Usage

### Running the Program

To start the program, run:

```bash
python main.py
```

### Menu Options

Upon execution, the program presents the following menu:

1. **Compress using SVD**: Enter the image path and perform SVD-based compression.
2. **Compress using PCA**: Enter the image path and perform PCA-based compression.
3. **Compress using NMF**: Enter the image path and perform NMF-based compression.
4. **Display Image Information**: Show details about the image.
5. **Resize Image**: Resize an image to a specified width and height.
6. **Crop Image**: Crop an image based on user input.
7. **Exit**: Close the program.

### Example Usage

#### Compress an Image Using SVD

1. Run the program.
2. Select option **1**.
3. Enter the path to the image file.
4. The program will perform SVD compression and display reconstructed images with different values of `k`.

## How It Works

### SVD Compression

1. Convert the image to grayscale.
2. Compute the SVD of the image matrix.
3. Reconstruct the image using the top `k` singular values.
4. Display the compressed image and compute the compression ratio.

### PCA Compression

1. Compute the covariance matrix of the image.
2. Perform eigenvalue decomposition.
3. Project the image onto the top `k` principal components.
4. Reconstruct the image and compute the compression ratio.

### NMF Compression

1. Factorize the image matrix into `W` and `H`.
2. Multiply `W` and `H` to reconstruct the image.
3. Compute the reconstruction error and compression ratio.

## Output

- The program displays the original image and reconstructed images for different values of `k`.
- Compression ratios and reconstruction errors are printed for analysis.

## Example Plots

- **Reconstruction Error vs k**
- **Compression Ratio vs k**

## License

This project is open-source and available under the MIT License.

## Author

Developed by Parikshit V

##Contact

For any inquiries, reach out at parikshitvel0210@gmail.com or open an issue on GitHub.


