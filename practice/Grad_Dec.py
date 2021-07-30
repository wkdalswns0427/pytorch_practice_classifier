import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

x_train = torch.FloatTensor([[73, 80, 75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180], [196], [142]])

class MultiVarReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self,x):
        return self.linear(x)

model = MultiVarReg()

'''
# 3 input 1 output
W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
'''

optimizer = optim.SGD(model.parameters(), lr=1e-5)

iteriations = 100
for epoch in range(iteriations+1):
    hypot = model(x_train)

    cost = F.mse_loss(hypot, y_train)
    #cost = torch.mean((hypot-y_train)**2)
    if epoch%10 == 0:
        print('epoch {:4d}/{} hypo: {} Cost: {:.6f}'.format(epoch, iteriations, hypot.squeeze().detach(), cost.item()))
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()