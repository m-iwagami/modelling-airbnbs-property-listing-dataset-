import itertools
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from joblib import load
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tabular_data import load_airbnb
from modelling_regression import split_data


class ClassificationModel:
    """
    A class for implementing and evaluating classification models for Airbnb property listing data.

    Attributes:
    - model: The classification model to be used (e.g., Logistic Regression, Decision Tree, etc.).

    Methods:
    - import_and_standarise_data(data_file): Load Airbnb data, handle missing values, and standardize numeric columns.
    - splited_data(X, y): Split data into training, validation, and test sets.
    - classification_metrics_performance(y_test, y_pred): Evaluate classification metrics (accuracy, precision, recall, F1 score).
    - confusion_matrix(model, X_test, y_test, y_pred): Generate and display the confusion matrix for the model.
    - tune_hyperparameters(model, X_train, X_val, X_test, y_train, y_val, y_test, hyperparameters): Perform hyperparameter tuning.
    - save_classification_model(folder_name, result_dict): Save the best classification model, hyperparameters, and metrics.
    - evaluate_model(model, hyperparameters_dict, folder_name): Evaluate a given classification model and save the results.
    """
    def __init__(self, model):
        """
        Initialize the ClassificationModel instance.

        Parameters:
        - model: The classification model to be used.
        """
        self.model  = model
        
    def import_and_standarise_data(self, data_file):
        """
        Load Airbnb data, handle missing values, and standardize numeric columns.

        Parameters:
        - data_file: The file containing Airbnb data.

        Returns:
        - X: Features (standardized and imputed).
        - y: Labels ("Category").
        """
        
       # Load the Airbnb data with "Category" as the label
        label_column = "Category"
        X, y = load_airbnb(data_file, label_column)

      # Select only numeric columns
        numeric_columns = X.select_dtypes(include=[np.number])
        df = pd.DataFrame(numeric_columns)
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        X = imputer.fit_transform(df)
        X = pd.DataFrame(X,y)
        return X, y
    
    def splited_data(self, X,y):
        """
        Split data into training, validation, and test sets.

        Parameters:
        - X: Features.
        - y: Labels.

        Returns:
        - X_train, X_validation, X_test: Features for training, validation, and testing.
        - y_train, y_validation, y_test: Labels for training, validation, and testing.
        """
        np.random.seed(10)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

    
        return X_train, X_validation, X_test, y_train, y_validation, y_test
                #70(train data)  15(validation) 15(test), 70(train data),15(validation) 15(test) ideal data split ratio
                #80(train data) 10(validation) 10(test) 80(train data) 10(validation) 10(test) 
                
    def classification_metrics_performance(self, y_test, y_pred):
        """
        Evaluate classification metrics (accuracy, precision, recall, F1 score).

        Parameters:
        - y_test: True labels.
        - y_pred: Predicted labels.

        Returns:
        - Dictionary containing accuracy, precision, recall, and F1 score.
        """
        #Scores from testing a model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")
        
        train_metrics = {
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1
            }
        return train_metrics
    
    def confusion_matrix(model, X_test, y_test, y_pred):
        """
        Generate and display the confusion matrix for the model.

        Parameters:
        - model: The classification model.
        - X_test: Features of the test set.
        - y_test: True labels of the test set.
        - y_pred: Predicted labels.

        Returns:
        - Confusion matrix display.
        """
        #confusion matrix for a model 
        title = "Confusion matrix"
        disp_1 = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        
        
        disp_2 = ConfusionMatrixDisplay.from_predictions(model, y_pred, y_test)
        disp_1.ax_.set_title(title)
        return disp_1

    def tune_hyperparameters(self, model,X_train, X_val, X_test, y_train, y_val, y_test, hyperparameters):
        """
        Perform hyperparameter tuning.

        Parameters:
        - model: The classification model.
        - X_train, X_val, X_test: Features for training, validation, and testing.
        - y_train, y_val, y_test: Labels for training, validation, and testing.
        - hyperparameters: Dictionary of hyperparameters to be tuned.

        Returns:
        - Dictionary containing the best model, hyperparameters, and evaluation metrics.
        """
        tuned_model = GridSearchCV(estimator=model, 
                               param_grid=hyperparameters, 
                               cv=5, 
                               scoring='accuracy')
        tuned_model.fit(X_train, y_train)
        

        best_hyperparameters = tuned_model.best_params_ 
        best_model_accuracy = tuned_model.best_score_
        best_classification_model = tuned_model.best_estimator_
        
        
        train_y_pred = best_classification_model.predict(X_train)
        valid_y_pred = best_classification_model.predict(X_val)
        y_pred = best_classification_model.predict(X_test)
        
        train_metrics = self.classification_metrics_performance(y_train, train_y_pred) 
        val_metrics = self.classification_metrics_performance(y_val, valid_y_pred)    
        test_metrics = self.classification_metrics_performance(y_test, y_pred)
                
        result_dict = {"Best_Model":best_classification_model, "Best_Hyperparameters":best_hyperparameters, "Best_Metrics": best_model_accuracy, "Train_Metrics": train_metrics, "validation_Metrics":val_metrics, "test_Metrics":test_metrics}
        return result_dict
        
    def save_classification_model(self, folder_name, result_dict):
        """
        Save the best classification model, hyperparameters, and metrics.

        Parameters:
        - folder_name: Name of the folder to save the model in.
        - result_dict: Dictionary containing the best model, hyperparameters, and metrics.

        Returns:
        - None
        """
        classification_dir = 'modelling-airbnbs-property-listing-dataset-/models/classification'
        current_dir = os.path.dirname(os.getcwd())
        folder_path = os.path.join(current_dir, classification_dir)
        folder_name_dir = os.path.join(folder_path, folder_name)
        
        if os.path.exists(folder_name_dir) == False:
            os.mkdir(folder_name_dir)

        best_model = list(result_dict.items())[0]
        best_hyperparameters = list(result_dict.items())[1]
        performance_metric = list(result_dict.items())[3:]

        # Save the model to a .joblib file
        joblib.dump(best_model, os.path.join(folder_name_dir,"model.joblib"))

        #Save the hyperparameters to a JSON file
        hyperparameters_filename = os.path.join(folder_name_dir, "hyperparameters.json")
        with open (hyperparameters_filename, 'w') as json_file:
            json.dump(best_hyperparameters, json_file)

        #Save the metrics to a JSON file
        metrics_filename = os.path.join(folder_name_dir, "metrics.json")
        with open (metrics_filename, 'w') as json_file:
            json.dump(performance_metric, json_file)

        return

    def evaluate_model(self, model, hyperparameters_dict, folder_name):
        """
        Evaluate a given classification model and save the results.

        Parameters:
        - model: The classification model.
        - hyperparameters_dict: Dictionary of hyperparameters for tuning.
        - folder_name: Name of the folder to save the results in.

        Returns:
        - None
        """
        df = "listing.csv"
        X, y = self.import_and_standarise_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = self.splited_data(X, y)
        result_dict = self.tune_hyperparameters(model, X_train, X_val, X_test, y_train, y_val, y_test, hyperparameters_dict)
        self.save_classification_model(folder_name, result_dict)

        return

def find_best_model(models):
    """
    Find the best classification model among a list of models.

    Parameters:
    - models: List of classification models.

    Returns:
    - Tuple containing the best model, hyperparameters, and evaluation metrics.
    """
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1": 0}
    
    clssification_dir = 'modelling-airbnbs-property-listing-dataset-/models/classification'
    current_dir = os.path.dirname(os.getcwd())
    folder_path = os.path.join(current_dir, clssification_dir)
        
    for i in range(len(models)):
        model_str = str(models[i])[0:-2]
        model_dir = os.path.join(folder_path, model_str)
        model = load(os.path.join(model_dir, 'model.joblib'))
        
        with open (os.path.join(model_dir, 'hyperparameters.json'),'r') as hyperparameters_path:
            hyperparameters = json.load(hyperparameters_path)
        
        with open(os.path.join(model_dir, 'metrics.json'), 'r') as metrics_path:
            metrics = json.load(metrics_path)
            metrics = metrics[2][1]
            
        if  best_hyperparameters_dict is None or round((metrics.get("Accuracy")),5) > best_metrics_dict.get("Accuracy"):
            best_regression_model = model
            best_hyperparameters_dict = hyperparameters
            best_metrics_dict = metrics
    return best_regression_model, best_hyperparameters_dict, best_metrics_dict

def logistic_regression(X,y):
    """
    Train a logistic regression model.

    Parameters:
    - X: Features.
    - y: Labels.

    Returns:
    - Tuple containing true labels, predicted labels, the trained model, and features for testing.
    """
    #Spliting Data
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=0)
    
    #training a model
    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    
    #Predicting
    y_pred = clf.predict(X_test)
    
    #accuracy score comparing the prediction and validation
    accuracy = accuracy_score(y_test, y_pred)
    return y_test, y_pred, clf, X_test


    


# Hyperparameters for different classification models
    
LogisticRegression_param = {
'C': [1.0],
'class_weight': ['balanced',None],
'dual': [True, False],
'fit_intercept': [True, False],
'intercept_scaling': [1],
'max_iter': [50, 100],
'multi_class': ['auto', 'ovr', 'multinomial'],
'n_jobs': [None],
'penalty': ['l1', 'l2', 'elasticnet', None],
'random_state': [None],
'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
'tol': [0.0001],
'verbose': [0],
'warm_start': [True, False]
}
DecisionTreeCLassifier_param = {
'max_depth': [1, 3, 5, None],
'min_samples_split': [3, 5, 10],
'random_state': [10, 20, None],
'splitter': ['best', 'random'],
  }
GardientBoostingClassifer_pram = {      
'loss': ['log_loss', 'exponential'],
'learning_rate' : [0.0, 0.1, 0.2],
'n_estimators': [100,200,300],
'criterion' : ['friedman_mse', 'squared_error']
  }
RandomForestClassifier_pram = {
'criterion':['gini', 'entropy', 'log_loss'],
'max_depth': [0.1, 1, 3, None],
'min_samples_leaf' : [1,2,3],
'max_features' : ['sqrt', 'log2', None]
  } 


# List of classification models

models =[   DecisionTreeClassifier(),
            LogisticRegression(),
            GradientBoostingClassifier(),
            RandomForestClassifier()]

if __name__ == "__main__":

    #Select model name    
    model = DecisionTreeClassifier()
    model = LogisticRegression()
    model = GradientBoostingClassifier()
    model = RandomForestClassifier()
    
    #Select folder name
    folder_name = 'DecisionTreeClassifier'
    folder_name = 'LogisticRegression'
    folder_name = 'GradientBoostingClassifier'
    folder_name = 'RandomForestClassifier'
    
    classification = ClassificationModel(model)
    classification.evaluate_model(model, DecisionTreeCLassifier_param, folder_name)
    
    find_best_model(models)


    