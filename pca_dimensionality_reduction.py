import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn import preprocessing

def load_and_plot_2d_data(file_path):
    """
    Load and visualize 2D data with PCA reduction to 1D.
    
    Parameters:
    file_path : str
        Path to the .mat file containing 2D data
        
    Returns:
    X : ndarray
        Original 2D data
    X_reconstructed : ndarray
        Reconstructed data from 1D PCA
    """
    # Load data
    data = loadmat(file_path)
    X = data['X']  # Shape: (m, 2)
    
    # Plot original data
    plt.figure()
    plt.plot(X[:, 0], X[:, 1], 'ro', label='Original Data')
    
    # Apply PCA to reduce to 1D
    pca = PCA(n_components=1)
    X1D = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X1D)
    
    # Plot reconstructed data
    plt.plot(X_reconstructed[:, 0], X_reconstructed[:, 1], 'b*', 
             label='Reconstructed from 1D')
    plt.title('PCA: 2D to 1D and Back')
    plt.legend()
    
    return X, X_reconstructed

def load_and_process_face_data(file_path):
    """
    Load face image data, normalize it, and apply PCA compression.
    
    Parameters:
    file_path : str
        Path to the .mat file containing face images
        
    Returns:
    data : ndarray
        Original face data
    data_norm : ndarray
        Normalized face data
    data_compressed : ndarray
        PCA-compressed data
    data_recovered : ndarray
        Reconstructed face data
    """
    # Load face data
    face_data = loadmat(file_path)
    data = face_data['X']  # Shape: (m, 1024) where 1024 = 32x32
    
    # Display first original face
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(data[0, :], (32, 32)).T, cmap='gray')
    plt.title('Original Face')
    
    # Normalize data
    scaler = preprocessing.StandardScaler().fit(data)
    data_norm = scaler.transform(data)
    
    # Apply PCA to compress (arbitrary choice of 324 components)
    pca_img = PCA(n_components=324)
    data_compressed = pca_img.fit_transform(data_norm)
    
    # Display first compressed face (reshaped to 18x18 for visualization)
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(data_compressed[0, :], (18, 18)).T, cmap='gray')
    plt.title('Compressed Face (324 components)')
    
    # Reconstruct data from compressed version
    data_recovered = pca_img.inverse_transform(data_compressed)
    
    # Display first reconstructed face
    plt.figure(figsize=(2, 2))
    plt.imshow(np.reshape(data_recovered[0, :], (32, 32)).T, cmap='gray')
    plt.title('Reconstructed Face')
    
    return data, data_norm, data_compressed, data_recovered

# Main execution
if __name__ == "__main__":
    # Part 1: 2D Data PCA
    data_path_2d = 'project8data1.mat'
    X, X_reconstructed = load_and_plot_2d_data(data_path_2d)
    
    # Part 2: Face Image PCA
    face_data_path = 'project8faces.mat'
    data, data_norm, data_compressed, data_recovered = load_and_process_face_data(face_data_path)
    
    # Show all plots
    plt.show()