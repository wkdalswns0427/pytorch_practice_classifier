import torch 
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data=([[73, 80, 75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]])
        self.y_data=([[152],[185],[180], [196], [142]])
    
    def __len__(self):
        #input data 크기
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x,y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size = 2, shuffle=True)

linear1 = torch.nn.Linear(3, 32, bias=True)
linear2 = torch.nn.Linear(32,16, bias=True)
linear3 = torch.nn.Linear(16,8, bias=True)
linear4 = torch.nn.Linear(8, 1, bias=True)
sigmoid = torch.nn.Sigmoid()

model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid, linear3, sigmoid, linear4, sigmoid)

#criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr =1)

iteriations = 10
for epoch in range(iteriations+1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        #prediction
        prediction = model(x_train)

        #cost
        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('epoch {:4d}/{} Batch: {}/{} Cost: {:.6f}'.format(epoch, iteriations, batch_idx+1, len(dataloader), cost.item()))
    
    