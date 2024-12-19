import pytest
import pandas as pd
import numpy as np
from src.datalib.manipulation_data import DataManipulation
import os
import tempfile

class TestDataManipulation:
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({
            'product_id': [101, 102, np.nan, 104],
            'product_name': ['Laptop', 'Smartphone', 'Tablet', np.nan],
            'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics'],
            'price': [999.99, np.nan, 499.99, 299.99],
            'stock_quantity': [50, 100, np.nan, 70]
        })
    
    def test_load_csv(self, sample_dataframe):
        """Test CSV loading functionality."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            sample_dataframe.to_csv(temp_file.name, index=False)
        
        # Load the CSV
        loaded_df = DataManipulation.load_csv(temp_file.name)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        # Assert loaded data matches original
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    def test_save_csv(self, sample_dataframe):
        """Test CSV saving functionality."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
            # Save DataFrame to CSV
            DataManipulation.save_csv(sample_dataframe, temp_file.name)
            
            # Reload and verify
            loaded_df = pd.read_csv(temp_file.name)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        # Assert loaded data matches original
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    def test_filter_data(self, sample_dataframe):
        """Test data filtering functionality."""
        # Filter by price condition
        filtered_df = DataManipulation.filter_data(
            sample_dataframe, 
            {'price': lambda x: x > 300}
        )
        
        # Expected result
        expected_df = sample_dataframe[sample_dataframe['price'] > 300]
        
        pd.testing.assert_frame_equal(filtered_df, expected_df)
    
    def test_handle_missing_values(self, sample_dataframe):
        """Test missing value handling."""
        # Test drop method: after dropping rows with NaN, expect 1 row remaining
        dropped_df = DataManipulation.handle_missing_values(sample_dataframe, method='drop')
        print("Dropped DataFrame:", dropped_df)  # Debug print to inspect the result
        assert len(dropped_df) == 1  # Expect 1 row remaining after dropping (verify based on NaN in the dataframe)
        
        # Test fill method: fill missing values with 0, no NaN values should remain
        filled_df = DataManipulation.handle_missing_values(sample_dataframe, method='fill', fill_value=0)
        assert filled_df.isna().sum().sum() == 0  # No missing values should remain

    def test_normalize_data(self, sample_dataframe):
        """Test data normalization."""
        normalized_df = DataManipulation.normalize_data(
            sample_dataframe, 
            columns=['price', 'stock_quantity']
        )
        
        # Check normalization ranges
        for column in ['price', 'stock_quantity']:
            assert normalized_df[column].min() >= 0
            assert normalized_df[column].max() <= 1
