import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear_layer = torch.nn.Linear(input_size, 1)

    def forward(self, features):
        return self.linear_layer(features)

    
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


def load_data():
    dataframe = pd.read_csv('clean_tabular_data.csv')
    X, y = dataframe.drop('Price_Night', axis=1), dataframe['Price_Night']
    return X,y

# Create dataset
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
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)




    
input_size = len(X.columns) 
model = LinearRegressionModel(input_size)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, dataloader, epochs=16):
    for epoch in range(epochs):
        for batch in dataloader:
            features, labels = batch
            optimizer.zero_grad()
            predictions = model(features)
            loss = F.mse_loss(predictions, labels)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(loss.item())

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
        
    
if __name__ == "__main__":
    model = NN()
    train(model, train_dataloader)