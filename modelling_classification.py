import numpy as np
import math
import pandas as pd
import itertools
import os
#import joblib
import json 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from modelling_regression import split_data
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from tabular_data import load_airbnb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay


def import_and_standarise_data(data_file):
    '''
        Imports the data through the load_airbnb() function and then standardises it

        Parameters
        ----------
        data_file

        Returns
        -------
        X: pandas.core.frame.DataFrame
            A pandas DataFrame containing the features of the model

        y: pandas.core.series.Series
            A pandas series containing the targets/labels 
      '''
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

  # Standadize numeric column
    #std = StandardScaler()
    #scaled_features = std.fit_transform(numeric_columns.values)
    #X = pd.DataFrame(scaled_features, index=numeric_columns.index, columns=numeric_columns.columns)
    #print(X)
    #Replace Null value with mean imputation
    #imputer = SimpleImputer(strategy='mean')
    #
    #X = imputer.fit_transform(X)
    #print(X)

def logistic_regression(X,y):
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

def splited_data(X,y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val, test_size=0.3, random_state=1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def classification_metrics_performance(y_test, y_pred, model, X_test):
    #Test
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    print("Accuracy_test: ", accuracy)
    print("Precision_test: ", precision)
    print("Recall_test: ", recall)
    print("F1_test: ", f1)
    
    title = "Confusion matrix"
    #confusion matric for a model 
    disp_1 = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    disp_2 = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp_1.ax_.set_title(title)
    print(title)
    print(disp_1.confusion_matrix)
    disp_2.ax_.set_title(title)
    print(disp_2.confusion_matrix)
    
    train_metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
    return train_metrics

def tune_classification_model_hyperparameters(models, X_train, X_val, X_test, y_train, y_val, 
                                              y_test, hyperparameters_dict):
    """
    """


    best_classification_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}
    
    for hyperparameter_values in itertools.product(*hyperparameters_dict):
        hyperparameters = dict(zip(hyperparameters_dict.keys(), hyperparameter_values))
        model = model.set_params(**hyperparameters)
        tuned_model = GridSearchCV(estimator=model, 
                               param_grid=hyperparameters, 
                               cv=5, 
                               scoring='accuracy')
        tuned_model = tuned_model.fit(X_train, y_train)

    best_hyperparameters_dict[model] = tuned_model.best_params_ 
    best_metrics_dict[model] = tuned_model.best_score_
    best_classification_model[model] = {'model': tuned_model.best_estimator_,
                                        'best_params': tuned_model.best_params_,
                                        'best_score': tuned_model.best_score_
                                          }

    if best_classification_model[model] is None or best_metrics_dict[model] > best_metrics_dict[best_classification_model]:
       best_classification_model = model
       best_hyperparameters = best_hyperparameters_dict[model]

    model = best_classification_model.fit(X_val,y_val)
    best_classification_model = model
    y_pred_test = best_classification_model.predict(X_test)

    best_metrics_dict = {
        'accuracy' : accuracy_score(y_test, y_pred_test),
        'precision' : precision_score(y_test, y_pred_test, average="macro"),
        'recall' : recall_score(y_test, y_pred_test, average="macro"),
        'f1' : f1_score(y_test, y_pred_test, average="macro")
    }
    print(best_classification_model, best_hyperparameters, best_metrics_dict)

def save_classification_model(folder_name, best_model, best_hyperparameters, performance_metric):
    """
        Save the best classification model, its hyperparameters, and performance metrics to files.

        Parameters:
            folder_name (str): The name of the folder to save the model.
            best_model: The best-trained regression model.
            best_hyperparameters (dict): A dictionary of the best hyperparameter values.
            performance_metric (dict): A dictionary of performance metrics.

        This function saves the best-trained classification model to a .joblib file, the best hyperparameters to a JSON file,
        and the performance metrics to another JSON file in the specified folder.

        Returns:
            None
    """
    models_dir = 'modelling-airbnbs-property-listing-dataset-/models'

    # Create Models folder
    current_dir = os.path.dirname(os.getcwd())
    models_path = os.path.join(current_dir, models_dir)
    if os.path.exists(models_path) == False:
        os.mkdir(models_path)

    # Create regression folder
    classification_dir = 'modelling-airbnbs-property-listing-dataset-/models/classification'
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, classification_dir)
    if os.path.exists(regression_path) == False:
        os.mkdir(regression_path)

    # Create linear_regression folder
    folder_name_dir = os.path.join(regression_path,folder_name)
    current_dir = os.path.dirname(os.getcwd())
    folder_name_path = os.path.join(current_dir, folder_name_dir)
    if os.path.exists(folder_name_path) == False:
        os.mkdir(folder_name_path)


    # Save the model to a .joblib file
    joblib.dump(best_model, os.path.join(folder_name_path,"model.joblib"))

    #Save the hyperparameters to a JSON file
    hyperparameters_filename = os.path.join(folder_name_path, "hyperparameters.json")
    with open (hyperparameters_filename, 'w') as json_file:
        json.dump(best_hyperparameters, json_file)
    
    #Save the metrics to a JSON file
    metrics_filename = os.path.join(folder_name_path, "metrics.json")
    with open (metrics_filename, 'w') as json_file:
        json.dump(performance_metric, json_file)

    return
    
hyperparameters_dict = [{ #LogisticRegression Hyperparameters
    'C': [1.0],
    'class_weight': ['balanced',None],
    'dual': [True, False],
    'fit_intercept': [True, False],
    'intercept_scaling': [1],
    'max_iter': [100],
    'multi_class': ['auto', 'ovr', 'multinomial'],
    'n_jobs': [None],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'random_state': [None],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'tol': [0.0001],
    'verbose': [0],
    'warm_start': [True, False]
},
  {#DecisionTreeCLassifier
'max_depth': [3, 10, 15, None],
'min_samples_split': [3, 20, 30],
'random_state': [None],
#'criterion':['gini', 'entropy', 'log_loss'],
'splitter': ['best', 'random'],
'max_featuresint': ['auto', 'sqrt', 'log2', None]
  },    
  {#GardientBoostingClassifer
      
'loss': ['log_loss', 'exponential'],
'learning_rate' : [0.0, 0.1, 0.2],
'n_estimators': [100,200,300],
'criterion' : ['friedman_mse', 'squared_error'],
'min_weight_fraction_leaf' : ['0.0', '0.1', '0.2','0.3'],
'max_features' : ['sqrt', 'log2', None]
  },
  {#RandomForestClassifier
      'criterion':['gini', 'entropy', 'log_loss'],
      'max_depth': [0.1, 1, 3, None],
      'min_samples_leafint' : [1,2,3],
      'max_features' : ['sqrt', 'log2', None]
  } 
]

models = [LogisticRegression(), DecisionTreeClassifier(),GradientBoostingClassifier(), RandomForestClassifier()]

if __name__ == "__main__":
    df = "listing.csv"
    X, y = import_and_standarise_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    #Logistic_regression confusion metrix
    #y_test, y_pred, model, X_test = logistic_regression(X,y)
    #classification_metrics_performance(y_test, y_pred, model, X_test)
    
    
    tune_classification_model_hyperparameters(models, X_train, X_val, X_test, y_train, y_val, y_test, hyperparameters_dict)
