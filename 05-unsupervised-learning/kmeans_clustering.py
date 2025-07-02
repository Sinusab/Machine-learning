import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Load the dataset from .mat file
data = loadmat('project8data1.mat')
X = data['X']  # Extract feature matrix (m samples, 2 features)

def find_closest_centroid(X, centroids):
    """
    Assign each data point to the closest centroid.
    
    Parameters:
    X : ndarray
        Data matrix of shape (m, n) where m is number of samples, n is number of features
    centroids : ndarray
        Current centroids of shape (K, n) where K is number of clusters
        
    Returns:
    idx : ndarray
        Array of shape (m,) containing cluster assignments (0 to K-1)
    """
    m = X.shape[0]  # Number of samples
    K = centroids.shape[0]  # Number of clusters
    
    idx = np.zeros(m, dtype=int)  # Array to store cluster assignments
    
    # For each data point
    for i in range(m):
        # Calculate Euclidean distance to each centroid
        distances = np.sqrt(np.sum(np.square(X[i] - centroids), axis=1))
        # Assign point to nearest centroid
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
    
    # Calculate mean for each cluster
    for k in range(K):
        centroids[k] = np.mean(X[idx == k], axis=0)
    
    return centroids

def k_means(X, initial_centroids, K, max_iter):
    """
    Run K-means clustering algorithm with visualization.
    
    Parameters:
    X : ndarray
        Data matrix of shape (m, n)
    initial_centroids : ndarray
        Initial centroids of shape (K, n)
    K : int
        Number of clusters
    max_iter : int
        Maximum number of iterations
        
    Returns:
    idx : ndarray
        Final cluster assignments
    centroids : ndarray
        Final centroids
    """
    centroids = initial_centroids.copy()
    
    # Initial plot of data points
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'ro', label='Data points')
    plt.title('Initial Data')
    
    # Main K-means loop
    for i in range(max_iter):
        # Assign points to clusters
        idx = find_closest_centroid(X, centroids)
        # Update centroids
        centroids = compute_centroids(X, idx, K)
        
        # Create new figure for each iteration
        plt.figure()
        
        # Plot centroids
        plt.plot(centroids[:, 0], centroids[:, 1], 'k*', 
                markersize=10, label='Centroids')
        
        # Plot points colored by cluster
        plt.plot(X[idx == 0, 0], X[idx == 0, 1], 'bo', label='Cluster 1')
        plt.plot(X[idx == 1, 0], X[idx == 1, 1], 'ro', label='Cluster 2')
        plt.plot(X[idx == 2, 0], X[idx == 2, 1], 'go', label='Cluster 3')
        
        plt.title(f'Iteration {i + 1}')
        plt.legend()
    
    return idx, centroids

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
    # Randomly permute indices
    rand_idx = np.random.permutation(m)
    # Select first K points as centroids
    centroids = X[rand_idx[:K], :]
    
    return centroids

# Main execution
if __name__ == "__main__":
    # Set parameters
    K = 3  # Number of clusters
    max_iter = 16  # Maximum iterations
    
    # Initial centroids
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    
    # Run K-means with initial centroids
    final_idx, final_centroids = k_means(X, initial_centroids, K, max_iter)
    
    # Get random centroids
    random_centroids = initialize_random_centroids(X, K)
    
    # Show all plots
    plt.show()