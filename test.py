from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('clean_tabular_data.csv')
        self.features_data = self.data.drop(columns=["Price_Night"])
        
    def __getitem__(self, index):
        label_data = self.data.iloc[index]
        feature_data = self.features_data.iloc[index]
        features = torch.tensor(feature_data[:]).float()
        labels = torch.tensor(label_data[3]).float()
        return features, labels

    
    def __len__(self):
        return len(self.data)

    def loader(self):
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        for batch in train_loader:
            features, labels = batch
            return features, labels         



class LinerRegression:
    def __init__(self) -> None:
        super().__init__()
        #initialise parameters
        self.linear_layer = torch.nn.Linear(11,10)
    
    def __call__(self, features):
        #use the layers to process the geatures
        return self.linear_layer(features)
    
    
dataset = AirbnbNightlyPriceRegressionDataset()
features, labels = dataset.loader()
features = features.reshape(16, -1)

model = LinerRegression()  
print(model(features))



