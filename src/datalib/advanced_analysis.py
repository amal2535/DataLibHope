from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier

class MachineLearningModels:
    
    @staticmethod
    def linear_regression(X, y):
        """
        Performs linear regression on the provided data.
        :param X: Features
        :param y: Target variable
        :return: Trained LinearRegression model
        """
        model = LinearRegression()
        model.fit(X, y)
        return model

    @staticmethod
    def kmeans_clustering(data, n_clusters):
        """
        Performs KMeans clustering on the data.
        :param data: The data to cluster
        :param n_clusters: The number of clusters to form
        :return: Trained KMeans model
        """
        model = KMeans(n_clusters=n_clusters)
        model.fit(data)
        return model

    @staticmethod
    def pca_analysis(data, n_components):
        """
        Performs PCA analysis on the data.
        :param data: The data for PCA
        :param n_components: Number of components to retain
        :return: Trained PCA model and transformed data
        """
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        return pca, transformed_data

    @staticmethod
    def decision_tree_classification(X, y):
        """
        Performs decision tree classification on the data.
        :param X: Features
        :param y: Target variable
        :return: Trained DecisionTreeClassifier model
        """
        model = DecisionTreeClassifier()
        model.fit(X, y)
        return model
