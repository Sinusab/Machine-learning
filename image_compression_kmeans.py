import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_image(image_path):
    """
    Load an image and preprocess it for clustering.
    
    Parameters:
    image_path : str
        Path to the image file
        
    Returns:
    image : ndarray
        Original image normalized to [0,1]
    img_flat : ndarray
        Flattened image array of shape (pixels, 3) for RGB
    original_shape : tuple
        Original dimensions of the image
    """
    # Load image
    image = mpl.image.imread(image_path)
    plt.figure()
    plt.imshow(image)
    plt.title("Original Image")
    
    # Normalize pixel values to [0,1]
    image = image / 255.0
    original_shape = image.shape
    
    # Reshape to (pixels, 3) for RGB clustering
    img_flat = image.reshape(-1, 3)
    
    return image, img_flat, original_shape

def find_closest_centroid(X, centroids):
    """
    Assign each data point to the closest centroid.
    
    Parameters:
    X : ndarray
        Data matrix of shape (m, n) where m is number of samples, n is features (RGB)
    centroids : ndarray
        Current centroids of shape (K, n) where K is number of clusters
        
    Returns:
    idx : ndarray
        Array of shape (m,) containing cluster assignments (0 to K-1)
    """
    m = X.shape[0]  # Number of samples
    K = centroids.shape[0]  # Number of clusters
    
    idx = np.zeros(m, dtype=int)
    
    # Compute distance of each point to all centroids
    for i in range(m):
        distances = np.sqrt(np.sum(np.square(X[i] - centroids), axis=1))
        idx[i] = np.argmin(distances)
    
    return idx

def compute_centroids(X, idx, K):
    """
    Compute new centroids based on current cluster assignments.
    
    Parameters:
    X : ndarray
        Data matrix of shape (m, n)
    idx : ndarray
        Cluster assignments of shape (m,)
    K : int
        Number of clusters
        
    Returns:
    centroids : ndarray
        Updated centroids of shape (K, n)
    """
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    # Calculate mean RGB values for each cluster
    for k in range(K):
        centroids[k] = np.mean(X[idx == k], axis=0)
    
    return centroids

def initialize_random_centroids(X, K):
    """
    Initialize centroids by randomly selecting K points from the dataset.
    
    Parameters:
    X : ndarray
        Data matrix of shape (m, n)
    K : int
        Number of clusters
        
    Returns:
    centroids : ndarray
        Randomly initialized centroids of shape (K, n)
    """
    m, n = X.shape
    # Randomly permute indices and select K points
    rand_idx = np.random.permutation(m)
    centroids = X[rand_idx[:K], :]
    
    return centroids

def k_means(X, initial_centroids, K, max_iter):
    """
    Run K-means clustering algorithm for image compression.
    
    Parameters:
    X : ndarray
        Flattened image data of shape (pixels, 3)
    initial_centroids : ndarray
        Initial centroids of shape (K, 3)
    K : int
        Number of clusters (colors)
    max_iter : int
        Maximum number of iterations
        
    Returns:
    idx : ndarray
        Final cluster assignments
    centroids : ndarray
        Final centroids (color palette)
    """
    centroids = initial_centroids.copy()
    
    # Run K-means iterations
    for i in range(max_iter):
        idx = find_closest_centroid(X, centroids)
        centroids = compute_centroids(X, idx, K)
    
    return idx, centroids

def compress_and_display_image(X, original_shape, K, max_iter):
    """
    Compress image using K-means and display the result.
    
    Parameters:
    X : ndarray
        Flattened image data of shape (pixels, 3)
    original_shape : tuple
        Original dimensions of the image
    K : int
        Number of colors for compression
    max_iter : int
        Maximum number of iterations
    """
    # Initialize random centroids
    initial_centroids = initialize_random_centroids(X, K)
    
    # Run K-means
    idx, centroids = k_means(X, initial_centroids, K, max_iter)
    
    # Reconstruct compressed image
    img_compressed = centroids[idx, :].reshape(original_shape)
    
    # Display compressed image
    plt.figure()
    plt.imshow(img_compressed)
    plt.title(f"Compressed Image (K={K}, Iterations={max_iter})")

# Main execution
if __name__ == "__main__":
    # Load and preprocess the image
    image_path = 'image.jpg'  # Replace with your image path
    image, img_flat, original_shape = load_and_preprocess_image(image_path)
    
    # Define compression levels to test
    compression_levels = [
        {"K": 4, "max_iter": 2},
        {"K": 4, "max_iter": 1},
        {"K": 2, "max_iter": 1},
        {"K": 8, "max_iter": 1},
        {"K": 16, "max_iter": 1}
    ]
    
    # Compress and display for each configuration
    for level in compression_levels:
        compress_and_display_image(img_flat, original_shape, 
                                level["K"], level["max_iter"])
    
    # Show all plots
    plt.show()