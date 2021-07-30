import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim as optim

train_losses = []
train_accs = []

valid_losses = []
valid_accs = []

trainset = dset.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download = True)
testset = dset.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download = True)

lr = 0.001
batch = 128
max_epoch = 20

dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch, shuffle = True, drop_last = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = batch, shuffle = False, drop_last = True)

for epoch in range(max_epoch):
    for X,Y in dataloader:
        X = X.view(-1,28,28)

print(torch.__version__)
#MNIST original size 28*28
# nn layers-batch normalization
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(28*28*1, 256, bias = True)
        self.linear2 =nn.Linear(256, 256, bias = True)
        self.linear3 =nn.Linear(256, 10, bias = True)
        #self.relu = F.ReLU()
        #self.bn1 = nn.BatchNorm1d(256)
        #self.bn2 = nn.BatchNorm1d(256)

    def forward(self,x):
       #x = x.view(-1,28*28*1) # 차원 처리
        x = self.linear1(x) # 레이어1
        #x = self.bn1(x) # batch nor
        x = F.relu(x)  # activation 
        x = self.linear2(x)
        #x = x = self.bn2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.log_softmax(x, dim =1)
        return X


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr =lr)

print(model)
train_total_batch = len(dataloader)
test_total_batch = len(testloader)

for epoch in range(max_epoch):
    model.train()
    avg_cost = 0
    
    for X, Y in dataloader:
        # reshape input image into [batch_size by 784]
        # label is not one-hot encoded
        X = X.view(-1, 28 * 28)
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / train_total_batch
        
    print("Epoch: ", "%04d" % (epoch+1), "cost =", "{:.9f}".format(avg_cost))

    with torch.no_grad():
        model.eval()
        loss, accuracy = 0, 0
        for i, (X,Y) in enumerate(dataloader):
            X = X.view(-1, 28 * 28)
            
            prediction = model(X)
            correct_prediction = torch.argmax(prediction, 1) == Y
            loss += criterion(prediction,Y)
            accuracy = correct_prediction.float().mean()

            loss = loss/train_total_batch
            
        train_losses.append([loss])
        train_accs.append([accuracy])

        print('[Epoch %d-TRAIN] Batchnorm Loss(Acc): loss:%.5f(accuracy:%.2f)' % ((epoch + 1), loss.item(), accuracy.item()))
        for i, (X,Y) in enumerate(testloader):
            X = X.view(-1, 28 * 28)
            
            prediction = model(X)
            correct_prediction = torch.argmax(prediction, 1) == Y
            loss += criterion(prediction,Y)
            accuracy = correct_prediction.float().mean()

            loss = loss/train_total_batch
            
        valid_losses.append([loss])
        valid_accs.append([accuracy])
        print('[Epoch %d-TEST] Batchnorm Loss(Acc): loss:%.5f(accuracy:%.2f)' % ((epoch + 1), loss.item(), accuracy.item()))
print("Accuracy: ", accuracy.item())
print()
print('Done')

plt.plot(train_losses)
plt.title("train losses")

plt.savefig("MNIST_LEARN.png")