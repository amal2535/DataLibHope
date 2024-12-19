import pytest
import pandas as pd
import numpy as np
from src.datalib.statistics import Statistics

class TestStatistics:

    def test_calculate_mean(self):
        """Test mean calculation."""
        data = [1, 2, 3, 4, 5]
        result = Statistics.calculate_mean(data)
        assert result == 3.0  # Mean of [1, 2, 3, 4, 5] is 3.0

    def test_calculate_median(self):
        """Test median calculation."""
        data = [1, 2, 3, 4, 5]
        result = Statistics.calculate_median(data)
        assert result == 3.0  # Median of [1, 2, 3, 4, 5] is 3.0

    def test_calculate_mode(self):
        """Test mode calculation."""
        data = pd.Series([1, 1, 2, 3, 4])
        result = Statistics.calculate_mode(data)
        assert result == 1  # Mode of [1, 1, 2, 3, 4] is 1

    def test_calculate_standard_deviation(self):
        """Test standard deviation calculation."""
        data = [1, 2, 3, 4, 5]
        result = Statistics.calculate_standard_deviation(data)
        expected_result = np.std(data)
        assert result == pytest.approx(expected_result, rel=1e-9)  # Allow for small floating-point differences

    def test_correlation_coefficient(self):
        """Test correlation coefficient calculation."""
        data1 = [1, 2, 3, 4, 5]
        data2 = [5, 4, 3, 2, 1]
        result = Statistics.correlation_coefficient(data1, data2)
        expected_result = np.corrcoef(data1, data2)[0, 1]
        assert result == pytest.approx(expected_result, rel=1e-9)  # Allow for small floating-point differences
