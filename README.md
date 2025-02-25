# MNIST Image Classifier with Dimensionality Reduction and k-NN
Project Overview
This project involves developing a robust classification system for the MNIST dataset. The system focuses on dimensionality reduction and k-Nearest Neighbors (k-NN) classification to predict handwritten digits. The solution was evaluated on clean, noisy, and masked datasets, demonstrating strong performance exceeding benchmarks.

Key Achievements
- Accuracy on Noisy Test Data: 93.1%
- Accuracy on Masked Test Data: 70.6%
- Effective use of PCA for dimensionality reduction and k-NN with weighted voting for classification.
- Systematic evaluation of preprocessing steps, classifier parameters, and feature selection.

Implementation Details
Dimensionality Reduction
- PCA: Used to reduce the 784-dimensional input space while retaining significant variance.
    - Reduced feature space effectively balances computational efficiency and accuracy.
    - The masked dataset experienced accuracy loss, suggesting further investigation into feature selection is needed.
Classification
- k-Nearest Neighbors (k-NN):
    - Weighted k-NN implemented with distance-based weights.
    - k=3 chosen for optimal performance, balancing sensitivity and robustness.
    - Larger k values were explored but showed diminishing accuracy returns.
Preprocessing
- Masked regions replaced with mean pixel values to mitigate occlusions.
- Potential future improvements include:
    - Using Gaussian filters or denoising methods for masked data.
    - Exploring different distance measures for enhanced classification.

Suggestions for Future Work
- Experiment with larger k values in k-NN.
- Compare performance across different distance measures in the classifier.
- Apply Gaussian filters or other denoising techniques to improve masked data handling.
- Explore advanced dimensionality reduction methods tailored for masked data.

How to Run
1. Train the Model:
    python train.py
- This generates a trained model saved as trained_model.pkl.
2. Evaluate the Model:
    python evaluate.py
- Outputs accuracy for noisy and masked datasets.

 
Note: This project was completed as part of a machine learning assignment focused on dimensionality reduction and classification of handwritten digit images.
