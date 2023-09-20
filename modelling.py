
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb


X, y = datasets.load_airbnb('clean_tabular_data.csv', "Price_Night")

print(f"Number of samples in dataset: {len(X)}")

def train_linear_regression_model(data_file):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5
    )

class LinearRegression:
    def __init__(self, n_features: int): # initalise parameters
        np.random.seed(10)
        self.W = np.random.randn(n_features, 1) ## randomly initialise weight
        self.b = np.random.randn(1) ## randomly initialise bias
        
    def __call__(self, X): # how do we calculate the output from an input in our model?
        ypred = np.dot(X, self.W) + self.b
        return ypred # return prediction
    
    def update_params(self, W, b):
        self.W = W ## set this instance's weights to the new weight value passed to the function
        self.b = b ## do the same for the bias


