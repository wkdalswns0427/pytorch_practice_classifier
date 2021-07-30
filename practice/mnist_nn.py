import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim as optim

trainset = dset.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download = True)
testset = dset.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download = True)

lr = 0.001
batch = 100
max_epoch = 20

dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch, shuffle = True, drop_last = True)

for epoch in range(max_epoch):
    for X,Y in dataloader:
        X = X.view(-1,28,28)

#MNIST original size 28*28
# nn layers-batch normalization
linear1 = nn.Linear(28*28*1, 256, bias = True)
linear2 = nn.Linear(256, 256, bias = True)
linear3 = nn.Linear(256, 10, bias = True)
relu = nn.ReLU()

nn.init.normal_(linear1.weight)
nn.init.normal_(linear2.weight)
nn.init.normal_(linear3.weight)

model = nn.Sequential(linear1, relu, linear2, relu, linear3)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr =lr)

total_batch = len(dataloader)

for epoch in range(max_epoch):
    model.train()
    avg_cost = 0
    total_batch = len(dataloader)
    for X, Y in dataloader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28)
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch
        
    print("Epoch: ", "%04d" % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

with torch.no_grad():
    model.eval()
    X_test = testset.test_data.view(-1, 28 * 28).float()
    Y_test = testset.test_labels
    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy: ", accuracy.item())

