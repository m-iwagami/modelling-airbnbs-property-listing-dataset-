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
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml




class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        assert len(X) == len(y)
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X.iloc[index].values, dtype=torch.float32), torch.tensor(self.y.iloc[index], dtype=torch.float32)

            
class NN(torch.nn.Module):
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


class LinearRegressionStructure(torch.nn.Module):#
    def __init__(self, config_model_structure):
        super().__init__()
        self.layers = nn.Sequential(
        config_model_structure
    )

    def forward(self, X):
        #return prediction 
        return self.layers(X)


class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_layer = torch.nn.Linear(input_size, 1)

    def forward(self, features):
        return self.linear_layer(features)

    

def get_model_structure(config):

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
def train(model, train_dataloader, val_dataloader, nn_config, epochs=16):

    lr = nn_config['lr']
    
    if nn_config['optimiser'] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr)
    elif nn_config['optimiser'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr)
    elif nn_config['optimiser'] == "SparseAdam":
        optimizer = torch.optim.SparseAdam(model.parameters(), lr)


    criterion = nn.MSELoss()
    writer = SummaryWriter()
    batch_idx = 0
    _batch_idx = 0
    
    starting_time = datetime.now()
    for epoch in range(epochs):
        for batch in train_dataloader:
            features, labels = batch
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar(('loss'), loss.item(), batch_idx)
            batch_idx += 1
            #print(f"Training Loss: {loss.item()}")

        prediction_time_list = []
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
            
    # getting the timestamp
    training_duration =  ending_time - starting_time
    time_filename = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    interference_latency =  sum(prediction_time_list) / len(prediction_time_list)

    print(model)
    #return model, training_duration, interference_latency, time_filename

def get_nn_config():
    with open('nn_config.yaml', 'r') as file:
        hyperparameter = yaml.safe_load(file)
    return hyperparameter

def generate_nn_config():
    
    combined_dictionary = {
    'optimiser': ['SGD', 'Adam', 'SparseAdam'],
    'lr': [0.01, 0.001, 0.0001],
    'hidden_layer_width': [16, 11, 10],
    'depth': [5, 3,1]
    }

    config_dict_list = []
    
    for iteration in itertools.product(*combined_dictionary.values()):
        config_dict = {
            'optimiser': iteration[0],
            'lr': iteration[1],
            'hidden_layer_width': iteration[2],
            'depth': iteration[3]}
        config_dict_list.append(config_dict)

    return config_dict_list
    
    

def find_best_nn(config_dict_list, train_dataloader, validation_dataloader, test_dataloader, model):
    
    # For each configuration, redefine the nn_model and the training function
    for i, nn_config in enumerate(config_dict_list):

        best_metrics_ = None
        best_hyperparameters = nn_config


        # Train the NN model using the model, the dataloaders and nn_config file
        best_model, training_duration, inference_latency, time_stamp = train(model, train_dataloader, validation_dataloader, nn_config)

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
        
        

    save_model(best_model_, best_hyperparameters_, best_metrics_, time_stamp)

    

    return best_metrics_, best_hyperparameters_
        
    

def calculate_metrics(best_model, train_loader, validation_loader, test_loader):
    def calculate_metrics_for_loader(loader):
        y = np.array([]) # Targets
        y_hat = np.array([])

        for features, labels in loader:
            features = features.to(torch.float32)  
            labels = labels.to(torch.float32).flatten()
            prediction = best_model(features).flatten()
            y = np.concatenate((y, labels.detach().numpy()))
            y_hat = np.concatenate((y_hat, prediction.detach().numpy()))
            # If the predictions include nan values, assign poor metrics to discard the model later
            if np.isnan(y_hat).any():
                RMSE_loss = 1000000
                R_squared = 0
            else:  # Else, calculate RMSE and R^2
                RMSE_loss = mean_squared_error(y, y_hat, squared=False)
                R_squared = r2_score(y, y_hat)

        return RMSE_loss, R_squared

    train_RMSE_loss, train_R_squared = calculate_metrics_for_loader(train_loader)
    validation_RMSE_loss, validation_R_squared = calculate_metrics_for_loader(validation_loader)
    test_RMSE_loss, test_R_squared = calculate_metrics_for_loader(test_loader)

    return train_RMSE_loss, validation_RMSE_loss, test_RMSE_loss, train_R_squared, validation_R_squared, test_R_squared

    #print(train_RMSE_loss, validation_RMSE_loss, test_RMSE_loss, train_R_squared, validation_R_squared, test_R_squared)
    

def save_model(model, best_hyperparameters_, best_metrics, time_stamp):
    dest = '/Users/momo/aicore/github/modelling-airbnbs-property-listing-dataset-/neural_networks/regression'
    new_path = f"{dest}/{time_stamp}"
    os.mkdir(new_path)
    
    #Save model
    torch.save(model.state_dict(), f'{new_path}/model.pt')
    
    #Save Hyperparameter
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
    dataframe = pd.read_csv('clean_tabular_data.csv')
    X, y = dataframe.drop('Price_Night', axis=1), dataframe['Price_Night']
    return X,y


X,y = load_data()
dataset = AirbnbNightlyPriceRegressionDataset(X, y)


# Split tarining, validation, testing samples (80,10,10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

#Dataset
train_dataset = AirbnbNightlyPriceRegressionDataset(X_train, y_train)
val_dataset = AirbnbNightlyPriceRegressionDataset(X_validation, y_validation)
test_dataset = AirbnbNightlyPriceRegressionDataset(X_test, y_test)


#DataLoader
batch_size = 16 
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    
if __name__ == "__main__":
    # Generate configurations
    config_dict_list = generate_nn_config()
    
    # Loop over each configuration
    for config in config_dict_list:
        #model = NN()
        # Get model structure based on the current configuration
        model_structure = get_model_structure(config)
        # Create LinearRegression_jared model using the model structure
        model = LinearRegressionStructure(model_structure)
        train(model, train_dataloader, val_dataloader, config)
        
    model, training_duration, interference_latency, time_filename = train(model, train_dataloader, val_dataloader, config)
    find_best_nn(config_dict_list, train_dataloader, val_dataloader, test_dataloader, model)
