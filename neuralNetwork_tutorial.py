import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##############신경망 정의#######################
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #입력 이미지 채널 1개, 출력 채널 6개, 3*3 컨볼류션 행렬 생성
        #커널 정의
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)

        self.fc1 = nn.Linear(16*6*6,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu( self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
#모델 내 학습 가능한 매개변수는 net.parameter()로 반환
'''
params = list(net.parameters())
print(len(params))
print(params[0].size())

#임의의 32*32 입력값
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
'''
#임의의 32*32 입력값
input = torch.randn(1, 1, 32, 32)

#########################손실함수########################
output = net(input)
target = torch.randn(10) #임의의 정답
target = target.view(1,-1)
criterion = nn.MSELoss() #mean-squared error 계산
loss = criterion(output, target)
print(loss)

#전체 그래프는 손실(loss)에 대하여 미분되며,
#그래프 내의 requires_grad=True 인 모든 Tensor는 변화도(gradient)가 누적된 .grad Tensor를 갖게 됩니다.
'''
print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) #Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLU
'''

############역전파###############################
#역전파 시 gradient 누적을 피하려면 0으로 맞춰줘야 함
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

###################가중치 갱신########################
optimizer = optim.SGD(net.parameters(), lr=0.01)

#training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() #업데이트