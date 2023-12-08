import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_diabetes

class DiabetesDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = load_diabetes(return_X_y=True)
         
    def __getitem__(self, index):
        return (torch.tensor(self.X[index]), torch.tensor(self.y[index]))
    
    def __len__(self):
        return (len(self.X))
        
dataset = DiabetesDataset()
print(dataset[10])
print(len(dataset))
        
#transform = transforms.PILToTensor()



#Dataset MNIST(handwritten letters)
#mnist_dataset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
#train_loader = DataLoader(mnist_dataset, batch_size=15, shuffle=True)
#for batch in train_loader:
#    print(batch)
#    features, labels = batch
#    print(features.shape)
#    print(labels.shape)
#    break

#print(label)

#float_tensor = torch.rand((4,2,3), dtype=torch.float32)
#double_tensor = float_tensor.double()
#print(f"random tensor {float_tensor.dtype}, {float_tensor.shape}")
#print(f"double{double_tensor}, shape: {double_tensor.shape}Type: {double_tensor.dtype}")


