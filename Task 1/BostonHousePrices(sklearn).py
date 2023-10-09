# Boston House Prices Prediction using sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Print the first 5 rows of the dataframe
print(boston_df.head()) 
# Print the keys of the dataset dictionary
print(boston.keys())

# then we know that the target variable is 'MEDV'
boston_df['MEDV'] = boston.target

x, test_x, y, test_y = train_test_split(boston.data, boston.target, test_size = 0.25, random_state = 40)

# Create a Linear regressor
LR = LinearRegression()

# Train the model using the training sets
LR.fit(x, y)
loss = LR.score(test_x, test_y)

print('loss:', loss)