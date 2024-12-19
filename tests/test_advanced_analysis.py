import unittest
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from src.datalib.advanced_analysis import MachineLearningModels

class TestMachineLearningModels(unittest.TestCase):

    def test_linear_regression(self):
        """Test linear regression."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = MachineLearningModels.linear_regression(X, y)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'coef_'))  # Check if model has coefficients
    
    def test_kmeans_clustering(self):
        """Test KMeans clustering."""
        X, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
        model = MachineLearningModels.kmeans_clustering(X, n_clusters=3)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'cluster_centers_'))  # Check if model has cluster centers

    def test_pca_analysis(self):
        """Test PCA analysis."""
        X, _ = make_blobs(n_samples=100, n_features=5, centers=3, random_state=42)
        pca, transformed_data = MachineLearningModels.pca_analysis(X, n_components=2)
        self.assertIsNotNone(pca)
        self.assertEqual(transformed_data.shape[1], 2)  # Check if data is reduced to 2 components

    def test_decision_tree_classification(self):
        """Test decision tree classification."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = MachineLearningModels.decision_tree_classification(X, y)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'feature_importances_'))  # Check if model has feature importances

if __name__ == "__main__":
    unittest.main()
