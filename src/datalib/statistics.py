import numpy as np
import pandas as pd

class Statistics:
    @staticmethod
    def calculate_mean(data):
        """Calculate the mean of a dataset."""
        return np.mean(data)

    @staticmethod
    def calculate_median(data):
        """Calculate the median of a dataset."""
        return np.median(data)

    @staticmethod
    def calculate_mode(data):
        """Calculate the mode of a dataset."""
        return data.mode()[0]

    @staticmethod
    def calculate_standard_deviation(data):
        """Calculate the standard deviation of a dataset."""
        return np.std(data)

    @staticmethod
    def correlation_coefficient(data1, data2):
        """Calculate the correlation coefficient between two datasets."""
        return np.corrcoef(data1, data2)[0, 1]
