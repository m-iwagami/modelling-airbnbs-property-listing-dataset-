
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tabular_data import load_airbnb

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


def train_linear_regression_model(data_file):
    # Load the Airbnb data with "Price_Night" as the label
    features, labels = load_airbnb(data_file, label_column="Price_Night")
    X, y = features, labels

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)
    X_validation, X_test, y_validation, y_test = model_selection.train_test_split(
    X_test, y_test, test_size=0.5
    )
   
    
    model = LinearRegression(n_features=18)  # instantiate our linear model
    y_pred = model(X_test)  # make predictions with data
    print("Predictions:\n", y_pred[:10]) # print the first 10 predictions
    
if __name__=="__main__":
    data_file = "clean_tabular_data.csv"
    train_linear_regression_model(data_file)



