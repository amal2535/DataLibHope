from src.datalib.manipulation_data import DataManipulation
import pandas as pd

# Load data
df = pd.DataFrame({'product_id': [101, 102, None], 'price': [999.99, None, 499.99]})

# Handle missing values
cleaned_df = DataManipulation.handle_missing_values(df, method='fill', fill_value=0)
print(cleaned_df)