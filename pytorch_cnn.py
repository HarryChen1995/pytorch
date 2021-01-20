#!/usr/bin/env python3
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms


train_dataset = datasets.FashionMNIST("./Desktop/", train = True, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]))
test_dataset = datasets.FashionMNIST("./Desktop/", train = False, download = True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]) )

train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size = 50 , shuffle = True)
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size = 50 , shuffle = True)

class  net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv1 = nn.Conv2d(1, 35, 4)
        self.cv2 = nn.Conv2d(35, 70, 3)
        self.cv3 = nn.Conv2d(70, 140, 2)
        x = torch.randn(28, 28).view(-1, 1, 28, 28)
        self._to_linear = None
        self.convs(x)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 10)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.cv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.cv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.cv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x =  F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim = 1)

cnn = net()
print(cnn)

onehot = lambda x: torch.eye(10,10)[x].tolist()
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
loss_func = nn.MSELoss()
epochs  = 1
for epoch in range(epochs):
    for data in train_dataset:
        cnn.zero_grad()
        x , y = data
        y = list(map(onehot, y))
        y = torch.tensor(y)
        output = cnn(x)
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch} : {loss.item()} loss")




correct = 0
total = 0

with torch.no_grad():
    for data in test_dataset:
        x, y = data
        output = cnn(x)
        total += 50
        temp = y  -  torch.argmax(output, dim = 1)
        temp = temp.size()[0] - torch.count_nonzero(temp).item()
        correct += temp
    accuracy = correct/total
    print(f"test accuracy { accuracy }")

label_map = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
for data in test_dataset:
    x, y = data
    print(f"predict item is {label_map[y[0].item()]}")
    plt.imshow(x[0].view(28,28).numpy())
    plt.show()
    break
