import torch, torchvision
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1,3,64,64) #3채널 64*64짜리 무작위 이미지 테이터
labels = torch.rand(1,1000)
#forward pass(순전파)
prediction = model(data)
loss = (prediction-labels).sum() #오차 텐서

#역전파 by 오차텐서.backward()
loss.backward()

#optimizer 모델
optim = torch.optim.SGD(model.parameters(),lr=1e-1,momentum=0.9)

optim.step()