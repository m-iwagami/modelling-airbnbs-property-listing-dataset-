# modelling.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from tabular_data import load_airbnb  # Import the load_airbnb function

def train_linear_regression_model(data_file):
    # Load the Airbnb data with "Price_Night" as the label
    features, labels = load_airbnb(data_file, label_column="Price_Night")
        
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    # Standardize the features (mean=0, std=1)
    X_test, X_validation, y_test, y_validation = train_test_split(
    X_test, y_test, test_size=0.3
)
    print(f"Number of samples in dataset: {len(features)}")
    check_nan = features.isnull().values.any()
    count_nan = features.isnull().sum()

    print(check_nan, count_nan)
    print("Number of samples in:")
    print(f"    Training: {len(y_train)}")
    print(f"    Testing: {len(y_test)}")


    np.random.seed(2)

    models = [
        DecisionTreeRegressor(splitter="random"),
        SVR(),
        LinearRegression()
    ]

    for model in models:
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_validation_pred = model.predict(X_validation)
        y_test_pred = model.predict(X_test)

        train_loss = mean_squared_error(y_train, y_train_pred)
        validation_loss = mean_squared_error(y_validation, y_validation_pred)
        test_loss = mean_squared_error(y_test, y_test_pred)

        print(
            f"{model.__class__.__name__}: "
            f"Train Loss: {train_loss} | Validation Loss: {validation_loss} | "
            f"Test Loss: {test_loss}"
        )
if __name__ == "__main__":
    data_file = "clean_tabular_data.csv"  # Replace with the actual file path
    train_linear_regression_model(data_file)
