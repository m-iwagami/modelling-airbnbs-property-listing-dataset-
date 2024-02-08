from datetime import datetime
import itertools
import json
import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml




class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, X, y):
        """
        Initializes the dataset.

        Args:
            X (pandas.DataFrame): Features.
            y (pandas.Series): Target variable.
        """
        super().__init__()
        assert len(X) == len(y)
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X.iloc[index].values, dtype=torch.float32), torch.tensor(self.y.iloc[index], dtype=torch.float32)

            
class NN(torch.nn.Module):
    """
    Neural network model for regression.

    This class defines a simple neural network architecture for regression tasks.

    Attributes:
        layers (torch.nn.Sequential): Layers of the neural network.

    """
    def __init__(self):
        
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,1)
        )


    def forward(self, X):
        #return prediction 
        return self.layers(X)


class LinearRegressionStructure(torch.nn.Module):
    """
    Linear regression model with custom structure.

    This class initializes a linear regression model with a custom structure defined by the user.

    Attributes:
        layers (torch.nn.Sequential): Layers of the linear regression model.

    """
    def __init__(self, config_model_structure):
        """
        Initializes the linear regression model with custom structure.

        Args:
            config_model_structure (OrderedDict): Configuration of the model structure.
        """
        
        super().__init__()
        self.layers = nn.Sequential(
        config_model_structure
    )

    def forward(self, X):
        #return prediction 
        return self.layers(X)


class LinearRegressionModel(torch.nn.Module):
    """
    Simple linear regression model.

    This class initializes a simple linear regression model.

    Attributes:
        linear_layer (torch.nn.Linear): Linear layer of the regression model.
    """

    def __init__(self, input_size):
        super().__init__()
        self.linear_layer = torch.nn.Linear(input_size, 1)

    def forward(self, features):
        return self.linear_layer(features)

def build_model_structure(config):
    """
    build_model_structure
    Constructs the model structure based on the provided configuration.

    Args:
        config (dict): Configuration of the model structure.

    Returns:
        OrderedDict: Model structure configuration.
    """
    hidden_layer_size = config['hidden_layer_width']
    linear_depth = config['depth']

    config_dict = OrderedDict()

    #input layer
    config_dict['input'] = nn.Linear(11, hidden_layer_size)

    for idx in range(linear_depth):
        rel_idx = f'relu{idx}'
        config_dict[rel_idx] = nn.ReLU()

        layer_idx = f'layer{idx}'
        config_dict[layer_idx] = nn.Linear(hidden_layer_size, hidden_layer_size)

    #outside the loop from here to add the
    config_dict[f'layer{linear_depth}'] = nn.Linear(hidden_layer_size, 11) 
    config_dict[f'relu{linear_depth}'] = nn.ReLU()

    #output layer
    config_dict['output'] = nn.Linear(11, 1)
    return config_dict


def train_neural_network(model, train_dataloader, val_dataloader, nn_config, epochs=40):
    
    """
    Trains the neural network model.

    Args:
        model (torch.nn.Module): Neural network model to be trained.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        nn_config (dict): Configuration for the neural network training.
        epochs (int, optional): Number of epochs. Defaults to 16.

    Returns:
        tuple: A tuple containing the trained model, training duration, interference latency, and timestamp.
    """
    
    lr = nn_config['lr']
    
    # Select optimizer based on the provided configuration
    if nn_config['optimiser'] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)
    elif nn_config['optimiser'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif nn_config['optimiser'] == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr)


    criterion = nn.MSELoss() # Define the loss function
    writer = SummaryWriter() # Tensorboard writer for logging
    batch_idx = 0
    _batch_idx = 0
    
    starting_time = datetime.now() # Start time for training duration calculation
    for epoch in range(epochs):
        for batch in train_dataloader:
            features, labels = batch
            optimizer.zero_grad() # Reset gradients
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward() # Backpropagation
            optimizer.step()
            writer.add_scalar(('loss'), loss.item(), batch_idx)
            batch_idx += 1
            #print(f"Training Loss: {loss.item()}")

        prediction_time_list = []
        # Validation phase
        for _batch in val_dataloader:
            features, labels =  _batch
            optimizer.zero_grad()  # Reset gradients before inference
            timer_start_ = time.time() # Start timer for interference_latency
            predictions = model(features)
            timer_end_ = time.time() # End timer for interference_latency
            batch_prediction_time = (timer_end_-timer_start_)/len(features) # Calculate interference_latency for each batch
            prediction_time_list.append(batch_prediction_time)
            loss = criterion(predictions, labels)
            writer.add_scalar(('loss_val'), loss.item(), _batch_idx)
            _batch_idx += 1

    #End of the training_duratioin time 
    ending_time = datetime.now()
    
    # Calculate training duration
    training_duration =  ending_time - starting_time
    
    # Generate timestamp
    time_filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    
    # Calculate average interference latency
    interference_latency =  sum(prediction_time_list) / len(prediction_time_list)

    return model, training_duration, interference_latency, time_filename

def get_nn_config():
    """
    Reads the neural network configuration from a YAML file.

    Returns:
        dict: Neural network configuration.
    """
    with open('nn_config.yaml', 'r') as file:
        hyperparameter = yaml.safe_load(file)
    return hyperparameter

def generate_nn_config():
    """
    Generates all possible configurations for neural network training.

    Returns:
        list: List of dictionaries containing different configurations.
    """
    
    # Define dictionaries for different hyperparameters
    combined_dictionary = {
    'optimiser': ['SGD', 'Adam', 'RMSProp'],
    'lr': [0.0005, 0.0008, 0.00001],
    'hidden_layer_width': [16, 11, 10],
    'depth': [5, 3, 1]
    }

    config_dict_list = []

    # Generate all combinations of hyperparameters
    for iteration in itertools.product(*combined_dictionary.values()):
        config_dict = {
            'optimiser': iteration[0],
            'lr': iteration[1],
            'hidden_layer_width': iteration[2],
            'depth': iteration[3]}
        config_dict_list.append(config_dict)

    return config_dict_list
    

def find_best_nn(config_dict_list, train_dataloader, validation_dataloader, test_dataloader):
    """
    Finds the best neural network configuration.

    Args:
        config_dict_list (list): List of dictionaries containing different configurations.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        validation_dataloader (DataLoader): DataLoader for the validation dataset.
        test_dataloader (DataLoader): DataLoader for the test dataset.

    Returns:
        tuple: A tuple containing the best metrics and hyperparameters.
    """    
    # Iterate over each configuration in the list
    for i, nn_config in enumerate(config_dict_list):
        
        # Build model structure based on the current configuration
        model_structure = build_model_structure(nn_config)
        
        # Create a model using the model structure
        model = LinearRegressionStructure(model_structure)
        
        # Train the neural network with the current configuration
        best_model, training_duration, inference_latency, time_stamp = train_neural_network(model, train_dataloader, val_dataloader, nn_config)

        best_metrics_ = None
        best_hyperparameters = nn_config

        # Calculate the metrics
        train_RMSE_loss, validation_RMSE_loss, test_RMSE_loss, train_R_squared, validation_R_squared, test_R_squared = calculate_metrics(best_model, train_dataloader, validation_dataloader, test_dataloader)

        best_metrics = {

        'RMSE_loss' : [train_RMSE_loss,validation_RMSE_loss,test_RMSE_loss],
        'R_squared' : [train_R_squared, validation_R_squared, test_R_squared],
        'training_duration' : training_duration,
        'inference_latency' : inference_latency,
    }
        # Store the metrics, config, and model:
        if best_metrics_ == None or best_metrics.get('R_squared')[1]>best_metrics_.get('R_squared')[1]:
            best_model_ = best_model
            best_hyperparameters_ = best_hyperparameters
            best_metrics_ = best_metrics

        if i >= 20:
            break
      
    # Save the best model, hyperparameters, and metrics  
    save_model(best_model_, best_hyperparameters_, best_metrics_, time_stamp)
    
    return best_metrics_, best_hyperparameters_
        
        
def calculate_metrics(best_model, train_loader, validation_loader, test_loader):
    """
    Calculates evaluation metrics for the trained model on different datasets.

    Args:
        best_model (torch.nn.Module): Trained neural network model.
        train_loader (DataLoader): DataLoader for the training dataset.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.

    Returns:
        tuple: A tuple containing the calculated RMSE loss and R-squared values for training, validation, and test datasets.
    """
    def calculate_metrics_for_loader(loader):
        y = np.array([]) 
        y_hat = np.array([])

        for features, labels in loader:
            features = features.to(torch.float32)  
            labels = labels.to(torch.float32).flatten()
            prediction = best_model(features).flatten()
            y = np.concatenate((y, labels.detach().numpy()))
            y_hat = np.concatenate((y_hat, prediction.detach().numpy()))
            # If the predictions include nan values, assign poor metrics to discard the model later
            if np.isnan(y_hat).any():
                RMSE_loss = 1000
                R_squared = 0
            else:  # Calculate RMSE and R^2
                RMSE_loss = mean_squared_error(y, y_hat, squared=False)
                R_squared = r2_score(y, y_hat)

        return RMSE_loss, R_squared
    
    # Calculate metrics for each loader (train, validation, test)
    train_RMSE_loss, train_R_squared = calculate_metrics_for_loader(train_loader)
    validation_RMSE_loss, validation_R_squared = calculate_metrics_for_loader(validation_loader)
    test_RMSE_loss, test_R_squared = calculate_metrics_for_loader(test_loader)

    return train_RMSE_loss, validation_RMSE_loss, test_RMSE_loss, train_R_squared, validation_R_squared, test_R_squared    

def save_model(model, best_hyperparameters_, best_metrics, time_stamp):
    """
    Saves the trained model, hyperparameters, and evaluation metrics to files.

    Args:
        model (torch.nn.Module): Trained neural network model.
        best_hyperparameters_ (dict): Best hyperparameters configuration.
        best_metrics (dict): Best evaluation metrics.
        time_stamp (str): Timestamp used for creating directories.
    """
    # Destination directory for saving model files
    dest = '/Users/momo/aicore/github/modelling-airbnbs-property-listing-dataset-/neural_networks/regression'
    new_path = f"{dest}/{time_stamp}"
    os.mkdir(new_path)
    
    # Save model state dictionary to a file
    torch.save(model.state_dict(), f'{new_path}/model.pt')
    
    # Save hyperparameters to a JSON file
    hyperparameter = {
        'optimiser': best_hyperparameters_['optimiser'],
        'lr': best_hyperparameters_['lr'],
        'hidden_layer_width': best_hyperparameters_['hidden_layer_width'],
        'depth': best_hyperparameters_['depth']
    }
    
    filename = os.path.join(new_path, "hyperparameters.json")
    with open(filename, 'w') as json_file:
        json.dump(hyperparameter, json_file)
    
    # Save the metrics to a JSON file
    training_duration = best_metrics['training_duration']
        
    metrics_data = {
    'RMSE_loss_train': best_metrics['RMSE_loss'][0],
    'RMSE_loss_validation': best_metrics['RMSE_loss'][1],
    'RMSE_loss_test': best_metrics['RMSE_loss'][2],
    'R_squared_train': best_metrics['R_squared'][0],
    'R_squared_validation': best_metrics['R_squared'][1],
    'R_squared_test': best_metrics['R_squared'][2],
    'training_duration_seconds': training_duration.total_seconds(),
    'inference_latency' : best_metrics['inference_latency']
    }

    #Save the metrics to a JSON file
    metrics_filename = os.path.join(new_path, "metrics.json")
    with open (metrics_filename, 'w') as json_file:
        json.dump(metrics_data, json_file)
        
    return
        
def load_data():
    """
    Load the Airbnb property listing data from 'clean_tabular_data.csv'.

    Returns:
    X (DataFrame): Features matrix containing all columns except 'Price_Night'.
    y (Series): Target variable 'Price_Night'.
    """
    dataframe = pd.read_csv('clean_tabular_data.csv')
    X, y = dataframe.drop('Price_Night', axis=1), dataframe['Price_Night']
    return X,y

def load_data_2():
    """
    Load the Airbnb property listing data from 'clean_tabular_data.csv'.

    Returns:
    X (DataFrame): Features matrix containing all columns except 'bedrooms'.
    y (Series): Target variable 'bedrooms'.
    """
    dataframe = pd.read_csv('clean_tabular_data.csv')
    X, y = dataframe.drop('bedrooms', axis=1), dataframe['bedrooms']
    return X,y

# Load data
X,y = load_data()
dataset = AirbnbNightlyPriceRegressionDataset(X, y)


# Split data into training, validation, and testing samples (80%, 10%, 10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

# Define datasets
train_dataset = AirbnbNightlyPriceRegressionDataset(X_train, y_train)
val_dataset = AirbnbNightlyPriceRegressionDataset(X_validation, y_validation)
test_dataset = AirbnbNightlyPriceRegressionDataset(X_test, y_test)


# Define dataloaders
batch_size = 16 
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    
if __name__ == "__main__":
    # Generate configurations
    config_dict_list = generate_nn_config()
    # Find the best neural network configuration
    find_best_nn(config_dict_list, train_dataloader, val_dataloader, test_dataloader)
