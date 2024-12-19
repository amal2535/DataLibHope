import pandas as pd
import pytest
import numpy as np
import matplotlib.pyplot as plt
from src.datalib.visualization import Plotting


class TestPlotting:
    
    def test_plot_histogram(self):
        """Test histogram plotting."""
        data = np.random.randn(1000)
        # Create the plot
        Plotting.plot_histogram(data, bins=20)
        # Close the plot after displaying
        plt.close()
        
    def test_plot_scatter(self):
        """Test scatter plot."""
        x = np.random.randn(100)
        y = np.random.randn(100)
        # Create the scatter plot
        Plotting.plot_scatter(x, y)
        # Close the plot after displaying
        plt.close()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample DataFrame for testing."""
        data = pd.DataFrame({
            'Category': ['A', 'B', 'C', 'D', 'E'],
            'Values': [23, 45, 56, 78, 33]
        })
        return data

    def test_plot_bar(self, sample_data):
        """Test plotting of bar chart."""
        Plotting.plot_bar(sample_data.set_index('Category')['Values'], title="Test Bar Chart")

    def test_plot_correlation_matrix(self, sample_data):
        """Test plotting of correlation matrix."""
        # Create a sample DataFrame with numeric data for correlation
        corr_data = pd.DataFrame({
            'A': [1, 2, 3, 4],
            'B': [5, 6, 7, 8],
            'C': [9, 10, 11, 12]
        })
        Plotting.plot_correlation_matrix(corr_data)
