from utils import *
import numpy as np
from scipy.ndimage import sobel
from scipy.spatial.distance import cdist

# Global variables for PCA
mean_vector = None
projection_matrix = None

# Preprocessing methods
def preprocess_masked_data(images):
    """
    Replace masked regions with the mean value of valid pixels in each image.

    Parameters:
        images (numpy.ndarray): Input image data.

    Returns:
        numpy.ndarray: Preprocessed image data.
    """
    for i in range(images.shape[0]):
        mask = images[i] != -1  # Identify valid pixels
        if np.any(mask):
            mean_value = np.mean(images[i][mask])
            images[i][~mask] = mean_value 
    return images

def preprocess_masked_data_with_edges(images):
    """
    Apply edge detection to preprocess masked image data.

    Parameters:
        images (numpy.ndarray): Input image data.

    Returns:
        numpy.ndarray: Preprocessed image data with edges highlighted.
    """
    processed_images = np.zeros_like(images)
    for i in range(images.shape[0]):
        mask = images[i] != -1  # Identify non-masked pixels
        if np.any(mask):
            mean_value = np.mean(images[i][mask])
            images[i][~mask] = mean_value 
        # Apply Sobel filter to highlight edges
        edge_image = sobel(images[i].reshape(28, 28), axis=-1)
        processed_images[i] = edge_image.flatten()
    return processed_images

def image_to_reduced_feature(images, split='train', n_components=None, use_edges=False):
    """
    Use raw pixel features or reduce dimensionality using PCA, optionally with edge detection.

    Parameters:
        images (numpy.ndarray): Input image data.
        split (str): Dataset split ('train', 'test', etc.).
        n_components (int): Number of components to keep for PCA.
        use_edges (bool): Whether to apply edge detection.

    Returns:
        numpy.ndarray: Feature vectors of shape (num_samples, num_features or n_components).
    """
    global mean_vector, projection_matrix

    # Step 1: Preprocess the masked data
    if use_edges:
        images = preprocess_masked_data_with_edges(images)
    else:
        images = preprocess_masked_data(images)

    if n_components is None:  # Use raw features
        return images

    # Step 2: PCA Implementation
    if split == 'train':
        # Calculate and save the mean vector
        mean_vector = np.mean(images, axis=0)
        centered_images = images - mean_vector

        # Compute the covariance matrix
        covariance_matrix = np.cov(centered_images, rowvar=False)

        # Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Select the top n_components eigenvectors
        top_indices = np.argsort(eigenvalues)[::-1][:n_components]
        projection_matrix = eigenvectors[:, top_indices]

        # Project training data onto the principal components
        reduced_features = np.dot(centered_images, projection_matrix)
    else:
        # For test data, reuse the mean vector and projection matrix from training
        centered_images = images - mean_vector
        reduced_features = np.dot(centered_images, projection_matrix)

    return reduced_features

# Classification: k-NN
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.train_features = None
        self.train_labels = None

    def fit(self, features, labels):
        """
        Store the training data for k-NN.

        Parameters:
            features (numpy.ndarray): Training features.
            labels (numpy.ndarray): Training labels.
        """
        self.train_features = features
        self.train_labels = labels

    def predict(self, test_features):
        """
        Predict labels for test data using k-NN.

        Parameters:
            test_features (numpy.ndarray): Test features.

        Returns:
            numpy.ndarray: Predicted labels.
        """
        # Compute pairwise distances between test and train data
        distances = cdist(test_features, self.train_features, metric='euclidean')

        # Find the indices of the k nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :self.k]

        # Select distances of the k nearest neighbors
        nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)

        # Determine the most common label among neighbors for each test instance
        nearest_labels = self.train_labels[nearest_indices]
        weights = 1 / (nearest_distances + 1e-9)  # Add small epsilon to avoid divide-by-zero
        predictions = np.array([np.bincount(labels, weights=weights).argmax() for labels, weights in zip(nearest_labels, weights)])

        return predictions

def training_model(train_features, train_labels):
    """
    Train the k-NN classifier.

    Parameters:
        train_features (numpy.ndarray): Training feature vectors.
        train_labels (numpy.ndarray): Corresponding training labels.

    Returns:
        KNNClassifier: Trained k-NN classifier.
    """
    knn = KNNClassifier(k=3) # Use k=3 as the default
    knn.fit(train_features, train_labels)
    return knn