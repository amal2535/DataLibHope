from src.datalib.manipulation_data import DataManipulation
from src.datalib.visualization import Plotting
from src.datalib.advanced_analysis import MachineLearningModels
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd

#tester sur iris database

# Charger les données depuis iris.csv
df = DataManipulation.load_csv("iris.csv")

# Gérer les valeurs manquantes avec DataManipulation
cleaned_df = DataManipulation.handle_missing_values(df, method='fill', fill_value=0)

# Afficher le DataFrame nettoyé
print("Cleaned DataFrame:")
print(cleaned_df.head())

# Tracer un histogramme pour 'SepalLengthCm'
if 'SepalLengthCm' in cleaned_df.columns:
    Plotting.plot_histogram(cleaned_df['SepalLengthCm'], bins=5, title="Histogram: Sepal Length")
    Plotting.plot_scatter(
        cleaned_df['SepalLengthCm'], 
        cleaned_df['SepalWidthCm'], 
        title="Scatter Plot: Sepal Length vs Width", 
        xlabel="Sepal Length (cm)", 
        ylabel="Sepal Width (cm)"
    )
else:
    print("La colonne 'SepalLengthCm' n'existe pas dans le DataFrame.")

# Préparer les données pour les modèles
features = cleaned_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
target = cleaned_df['Species'] if 'Species' in cleaned_df.columns else None

if target is not None:
    # Encodage des étiquettes de classe si nécessaire
    target = pd.factorize(target)[0]

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Test du modèle Linear Regression
    lin_reg_model = MachineLearningModels.linear_regression(X_train, y_train)
    print(f"Linear Regression Coefficients: {lin_reg_model.coef_}")

    # Test du modèle KMeans
    kmeans_model = MachineLearningModels.kmeans_clustering(features, n_clusters=3)
    print(f"KMeans Cluster Centers:\n{kmeans_model.cluster_centers_}")

    # Test de PCA
    pca, transformed_data = MachineLearningModels.pca_analysis(features, n_components=2)
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

    # Test du modèle Decision Tree
    decision_tree_model = MachineLearningModels.decision_tree_classification(X_train, y_train)
    predictions = decision_tree_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")
else:
    print("La colonne 'Species' n'existe pas dans le DataFrame.")

# Fin du test
print("Tests complets.")