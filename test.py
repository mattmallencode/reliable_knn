import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("datasets/abalone.data")

new_column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]


df.columns = new_column_names
df["Age"] = df["Rings"]


df.drop('Sex', axis=1, inplace=True)

df.drop('Rings', axis=1, inplace=True)



X = df.drop(columns=['Age']) # Features without the ‘Age’ column
y = df['Age'] # Target column (‘Age’)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = KNeighborsRegressor(n_neighbors=29, weights='distance')

regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print(mae)