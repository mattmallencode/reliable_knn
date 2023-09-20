import pandas as pd
import numpy as np
from pandas import DataFrame, Series

# Read csv into DF.
data: DataFrame = pd.read_csv('datasets/processed.cleveland.data')

# Replacing missing data string with np.nan.
data.replace(["?"], np.nan, inplace=True)

# Drop rows containing NaN.
data.dropna(inplace=True)
data = data.astype(float)

# Replace values greater than 0 with 1 in the last column.
data.iloc[:, -1]: Series = data.iloc[:, -1].apply(lambda x: 1 if x > 0 else 0)

# Save the modified data to a new file.
data.to_csv('datasets/custom_cleveland.data', index=False)