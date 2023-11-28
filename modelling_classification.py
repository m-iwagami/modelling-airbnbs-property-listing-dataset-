import numpy as np
import math
import pandas as pd
import itertools
import os
import joblib
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

class ClassificationModel:
    def __init__(self, model):
        self.model  = model
        
    def import_and_standarise_data(self, data_file):
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
    
    def splited_data(self, X,y):
        np.random.seed(10)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
        X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

    
        return X_train, X_validation, X_test, y_train, y_validation, y_test
                #70(train data)  15(validation) 15(test), 70(train data),15(validation) 15(test) ideal data split ratio
                #80(train data) 10(validation) 10(test) 80(train data) 10(validation) 10(test) 
                
    def classification_metrics_performance(self, y_test, y_pred):
    
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
        #confusion matrix for a model 
        title = "Confusion matrix"
        disp_1 = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        
        
        disp_2 = ConfusionMatrixDisplay.from_predictions(model, y_pred, y_test)
        disp_1.ax_.set_title(title)
        return disp_1

    def tune_hyperparameters(self, model,X_train, X_val, X_test, y_train, y_val, y_test, hyperparameters):
        
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
                
        #nl = '\n'        
        #print(f"Best Model: {best_classification_model}, {nl} Best Hyperparameters: {best_hyperparameters}, {nl} Best Metrics: {best_model_accuracy}, {nl} Train Metrics: {train_metrics}, {nl} validation Metrics: {val_metrics}, {nl} test Metrics: {test_metrics}")
        
        result_dict = {"Best_Model":best_classification_model, "Best_Hyperparameters":best_hyperparameters, "Best_Metrics": best_model_accuracy, "Train_Metrics": train_metrics, "validation_Metrics":val_metrics, "test_Metrics":test_metrics}
        return result_dict

    def save_model(self, result_dict):
        model_dir = "/Users/momo/aicore/github/modelling-airbnbs-property-listing-dataset-/models/classification"
        
    def save_classification_model(self, folder_name, result_dict):

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
DecisionTreeCLassifier_param = {#DecisionTreeCLassifier
'max_depth': [1, 3, 5, None],
'min_samples_split': [3, 5, 10],
'random_state': [10, 20, None],
#'criterion':['gini', 'entropy', 'log_loss'],
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



if __name__ == "__main__":
    #model = DecisionTreeClassifier()
    #model = LogisticRegression()
    #model = GradientBoostingClassifier()
    model = RandomForestClassifier()
    
    
    classification = ClassificationModel(model)
    df = "listing.csv"
    X, y = classification.import_and_standarise_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = classification.splited_data(X, y)
    #result_dict = classification.tune_hyperparameters(model, X_train, X_val, X_test, y_train, y_val, y_test, DecisionTreeCLassifier_param)
    #result_dict = classification.tune_hyperparameters(model, X_train, X_val, X_test, y_train, y_val, y_test, LogisticRegression_param)
    #result_dict = classification.tune_hyperparameters(model, X_train, X_val, X_test, y_train, y_val, y_test, GardientBoostingClassifer_pram)
    result_dict = classification.tune_hyperparameters(model, X_train, X_val, X_test, y_train, y_val, y_test, RandomForestClassifier_pram)

    #folder_name = 'DecisionTreeClassifier'
    #folder_name = 'LogisticRegression'
    #folder_name = 'GradientBoostingClassifier'
    folder_name = 'RandomForestClassifier'
    
    classification.save_classification_model(folder_name, result_dict)


    