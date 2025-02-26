from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to recreate image using top k singular values
def recreate_approx_image(u, sigma, v, k):
    B = np.dot(u[:, :k], np.dot(np.diag(sigma[:k]), v[:k, :]))
    A_hat = np.clip(B, 0, 255).astype(np.uint8)
    return A_hat

# Function to calculate reconstruction error
def get_error(A, A_hat):
    return np.linalg.norm(A - A_hat)

# Function for SVD-based image compression
def svd_compression(image_path):
    print("~~Image Compression Using SVD~~")
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()
    img = img.convert('L')  # Convert image to grayscale
    A = np.array(img)

    u, sigma, v = np.linalg.svd(A, full_matrices=False)
    m, n = A.shape
    print('Dimensions of the Image:')
    print("row =", m)
    print("col =", n)

    errors = []
    compression_ratios = []
    for k in [2, 4, 8, 16, 32, 64]:
        A_hat = recreate_approx_image(u, sigma, v, k)
        error = get_error(A, A_hat)
        errors.append(error)
       
        print(f"At k = {k}:")
        print("Reconstruction Error:", round(error, 4))
       
        plt.subplot(1, 2, 1)
        plt.imshow(A, cmap='gray')
        plt.title('Original Image')
       
        plt.subplot(1, 2, 2)
        plt.imshow(A_hat, cmap='gray')
        plt.title(f'Reconstructed Image (k={k})')
       
        plt.show()

        full_representation = m * n
        svd_rep = k * (m + n + 1)
        compression_ratio = svd_rep / full_representation
        compression_ratios.append(compression_ratio)
        print("Compression ratio:", round(compression_ratio, 5))
        print('\n')

    plt.plot([2, 4, 8, 16, 32, 64], errors)
    plt.title('Reconstruction Error vs k')
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.show()

    plt.plot([2, 4, 8, 16, 32, 64], compression_ratios)
    plt.title('Compression Ratio vs k')
    plt.xlabel('k')
    plt.ylabel('Compression Ratio')
    plt.show()

# Function to perform manual PCA and reconstruct the image
def manual_pca_reconstruction(A, k):
    A_meaned = A - np.mean(A, axis=0)
    covariance_matrix = np.cov(A_meaned, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
   
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
   
    W = sorted_eigenvectors[:, :k]
   
    A_reduced = np.dot(A_meaned, W)
   
    A_hat = np.dot(A_reduced, W.T) + np.mean(A, axis=0)
   
    A_hat = np.clip(A_hat, 0, 255).astype(np.uint8)
   
    return A_hat

# Function for PCA-based image compression
def pca_compression(image_path):
    print("~~Image Compression Using Manual PCA~~")
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()
   
    img = img.convert('L')  # Convert image to grayscale
    A = np.array(img)

    m, n = A.shape
    print('Dimensions of the Image:')
    print("row =", m)
    print("col =", n)

    errors = []
    compression_ratios = []
    for k in [2, 4, 8, 16, 32, 64]:
        A_hat = manual_pca_reconstruction(A.astype(float), k)
        error = get_error(A.astype(float), A_hat)
        errors.append(error)

        print(f"At k = {k}:")
        print("Reconstruction Error:", round(error, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(A.astype(float), cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(A_hat.astype(float), cmap='gray')
        plt.title(f'Reconstructed Image (PCA k={k})')

        plt.show()

        full_representation = m * n
        pca_rep = k * (m + n)
        compression_ratio = pca_rep / full_representation
        compression_ratios.append(compression_ratio)
        print("Compression ratio (PCA):", round(compression_ratio, 5))
        print('\n')

    plt.plot([2, 4, 8, 16, 32, 64], errors)
    plt.title('Reconstruction Error vs k')
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.show()

    plt.plot([2, 4, 8, 16, 32, 64], compression_ratios)
    plt.title('Compression Ratio vs k')
    plt.xlabel('k')
    plt.ylabel('Compression Ratio')
    plt.show()

# Function to perform manual NMF and reconstruct the image
def manual_nmf_reconstruction(A, k, max_iter=500, tol=1e-4):
    # Initialize W and H matrices with random non-negative values
    m, n = A.shape
    W = np.random.rand(m, k)
    H = np.random.rand(k, n)

    # NMF algorithm
    for _ in range(max_iter):
        # Update H
        H *= (W.T @ A) / (W.T @ W @ H + 1e-10)  # Avoid division by zero
       
        # Update W
        W *= (A @ H.T) / (W @ H @ H.T + 1e-10)  # Avoid division by zero
       
        # Check for convergence
        if np.linalg.norm(A - W @ H) < tol:
            break

    # Reconstruct the image
    A_hat = np.dot(W, H)
   
    # Ensure pixel values are between 0 and 255
    A_hat = np.clip(A_hat, 0, 255).astype(np.uint8)
   
    return A_hat

# Function for NMF-based image compression
def nmf_compression(image_path):
    print("~~Image Compression Using Manual NMF~~")
   
    img = Image.open(image_path)
    plt.imshow(img)
    plt.show()

    img = img.convert('L')  # Convert image to grayscale
    A = np.array(img)

    m, n = A.shape
    print('Dimensions of the Image:')
    print("row =", m)
    print("col =", n)

    errors = []
    compression_ratios = []
    for k in [2, 4, 8, 16]:  # Number of components to keep (NMF typically uses fewer components than PCA/SVD)
        A_hat = manual_nmf_reconstruction(A.astype(float), k)
        error = get_error(A.astype(float), A_hat)
        errors.append(error)

        print(f"At k = {k}:")
        print("Reconstruction Error:", round(error, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(A.astype(float), cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(A_hat.astype(float), cmap='gray')
        plt.title(f'Reconstructed Image (NMF k={k})')

        plt.show()

        full_representation = m * n
        nmf_rep = k * (m + n)   # Number of parameters in W and H
       
        # Compression ratio for NMF
        compression_ratio = nmf_rep / full_representation
        compression_ratios.append(compression_ratio)
        print("Compression ratio (NMF):", round(compression_ratio, 5))
        print('\n')

    plt.plot([2, 4, 8, 16], errors)
    plt.title('Reconstruction Error vs k')
    plt.xlabel('k')
    plt.ylabel('Reconstruction Error')
    plt.show()

    plt.plot([2, 4, 8, 16], compression_ratios)
    plt.title('Compression Ratio vs k')
    plt.xlabel('k')
    plt.ylabel('Compression Ratio')
    plt.show()

# Function to display image information
def display_image_info(image_path):
    img = Image.open(image_path)
    print("Image Information:")
    print("Format:", img.format)
    print("Size:", img.size)
    print("Mode:", img.mode)

    plt.imshow(img)
    plt.title('Original Image')
    plt.show()

# Function to resize an image
def resize_image(image_path, new_size):
    img = Image.open(image_path)
    img = img.resize(new_size)
    img.save("resized_image.jpg")
    print("Image resized and saved as 'resized_image.jpg'")

    plt.imshow(img)
    plt.title('Resized Image')
    plt.show()

# Function to crop an image
def crop_image(image_path, crop_box):
    img = Image.open(image_path)
    img = img.crop(crop_box)
    img.save("cropped_image.jpg")
    print("Image cropped and saved as 'cropped_image.jpg'")

    plt.imshow(img )
    plt.title('Cropped Image')
    plt.show()

# Main menu function
def main_menu():
    print("~ Image Compression Menu ~")
   
    while True:
        print("1. Compress using SVD")
        print("2. Compress using PCA")
        print("3. Compress using NMF")
        print("4. Display Image Information")
        print("5. Resize Image")
        print("6. Crop Image")
        print("7. Exit")
       
        choice = input("Enter your choice (1/2/3/4/5/6/7): ")
       
        if choice == '1':
            image_path = input("Enter the path of the image file: ")
            svd_compression(image_path)
       
        elif choice == '2':
            image_path = input("Enter the path of the image file: ")
            pca_compression(image_path)

        elif choice == '3':
            image_path = input("Enter the path of the image file: ")
            nmf_compression(image_path)

        elif choice == '4':
            image_path = input("Enter the path of the image file: ")
            display_image_info(image_path)

        elif choice == '5':
            image_path = input("Enter the path of the image file: ")
            new_size = tuple(map(int, input("Enter the new size (width height): ").split()))
            resize_image(image_path, new_size)

        elif choice == '6':
            image_path = input("Enter the path of the image file: ")
            crop_box = tuple(map(int, input("Enter the crop box (left top right bottom): ").split()))
            crop_image(image_path, crop_box)

        elif choice == '7':
            print("Exiting...")
            break
       
        else:
            print("Invalid choice. Please enter a valid option.")

# Run the main menu function
if __name__ == "__main__":
    main_menu()