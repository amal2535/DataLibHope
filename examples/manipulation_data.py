from src.datalib.manipulation_data import DataManipulation
from src.datalib.visualization import Plotting
import pandas as pd

#tester sur iris database
# Charger les données depuis iris.csv
df = DataManipulation.load_csv("iris.csv")

# Gérer les valeurs manquantes avec DataManipulation
cleaned_df = DataManipulation.handle_missing_values(df, method='fill', fill_value=0)

# Afficher le DataFrame nettoyé
print(cleaned_df)

# Tracer un histogramme pour une colonne numérique, par exemple 'sepal_length'
if 'SepalLengthCm' in cleaned_df.columns:
    Plotting.plot_histogram(cleaned_df['SepalLengthCm'], bins=5)
    Plotting.plot_scatter(cleaned_df['SepalLengthCm'], cleaned_df['SepalWidthCm'])
else:
    print("La colonne 'sepal_length' n'existe pas dans le DataFrame.")
