from typing import Optional, Union
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataManipulation:
    @staticmethod
    def load_csv(filepath):
        """
        Charge un fichier CSV et retourne un DataFrame pandas.
        :param filepath: Le chemin du fichier CSV à charger.
        :return: pandas.DataFrame
        """
        return pd.read_csv(filepath)

    @staticmethod
    def save_csv(dataframe, filepath):
        """
        Enregistre un DataFrame pandas dans un fichier CSV.
        :param dataframe: Le DataFrame à enregistrer.
        :param filepath: Le chemin du fichier de sortie.
        """
        dataframe.to_csv(filepath, index=False)

    @staticmethod
    def filter_data(dataframe, condition):
        """
        Filtre les données d'un DataFrame selon une condition.
        :param dataframe: Le DataFrame à filtrer.
        :param condition: Un dictionnaire de conditions à appliquer sur les colonnes.
        :return: pandas.DataFrame
        """
        for column, func in condition.items():
            dataframe = dataframe[func(dataframe[column])]
        return dataframe

    @staticmethod
    def normalize_data(dataframe, columns):
        """
        Normalise les données des colonnes spécifiées entre 0 et 1.
        :param dataframe: Le DataFrame contenant les données.
        :param columns: Liste des noms des colonnes à normaliser.
        :return: pandas.DataFrame avec les données normalisées.
        """
        scaler = MinMaxScaler()
        dataframe[columns] = scaler.fit_transform(dataframe[columns])
        return dataframe


    @staticmethod
    def handle_missing_values(df, method='fill', fill_value=None):
        """
        Handle missing values in the DataFrame.
        :param df: The DataFrame with missing values.
        :param method: The method to handle missing values. Can be 'drop', 'fill'.
        :param fill_value: The value to use for filling missing values (used when method='fill').
        :return: DataFrame with missing values handled.
        """
        if method == 'drop':
            # Drop rows with any missing values
            return df.dropna(how='any')  # Explicitly drop rows with any NaN values
        elif method == 'fill':
            # Fill missing values with a specific fill_value
            return df.fillna(fill_value)
        else:
            raise ValueError("Method must be 'drop' or 'fill'.")