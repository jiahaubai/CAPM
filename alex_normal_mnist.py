import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.utils.data as Data
from torchvision import transforms
import pandas as pd
import torch.optim as optim

from utils_mnist import (upper_limit, lower_limit, std, clamp, get_loaders,
    evaluate_pgd, evaluate_standard, evaluate_fgsm)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) #--------------------------------------------
])

transform_test = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,)) #--------------------------------------------
])

print('Downloading MNIST')
train_data = torchvision.datasets.MNIST(
    root='./mnist', train=True, download=True, transform=transform)
print('Downloaded')

# split validation set
torch.manual_seed(1219)
val_size = 5000
train_size = len(train_data) - val_size
train_ds, val_ds = random_split(train_data, [train_size, val_size])

# use same dataset (first 100 of Cifar10 test)
df = pd.read_csv('../data/mnist_test.csv')
data = df.iloc[:100, 1:].values.astype(float)
label = df.iloc[:100, 0].values

data_tensor = torch.from_numpy(data).float().view(
    len(data), -1, 28, 28)  # -------------------------------------------
target_tensor = torch.from_numpy(label).long()


class testset(Dataset):
    def __init__(self, data_tensor, target_tensor, loader=transform_test):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.loader = loader

    def __getitem__(self, index):
        data = self.data_tensor[index] / 255
        data = self.loader(data)
        target = self.target_tensor[index]
        return data, target

    def __len__(self):
        return self.data_tensor.size()[0]


test_ds = testset(data_tensor, target_tensor)

batch_size = 4
trainloader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2)
valloader = DataLoader(val_ds, batch_size, num_workers=2)
testloader = DataLoader(
    test_ds, shuffle=False, batch_size=4, num_workers=2)

print('Succefully load data, size of train, val, test are',
      len(train_ds), len(val_ds), len(test_ds))

# Use the AlexNet Pretrained model
AlexNet_Model = torch.hub.load(
    'pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
AlexNet_Model.eval()

AlexNet_Model.features[0] = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
AlexNet_Model.features[3] = nn.Conv2d(8, 24, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
AlexNet_Model.features[6] = nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
AlexNet_Model.features[8] = nn.Conv2d(48, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
AlexNet_Model.features[10] = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


AlexNet_Model.classifier[1] = nn.Linear(288, 512)
AlexNet_Model.classifier[4] = nn.Linear(512, 512)
AlexNet_Model.classifier[6] = nn.Linear(512, 10)
tmp0 = nn.ModuleList(AlexNet_Model.children())[0][0:8]
tmp1 = nn.ModuleList(AlexNet_Model.children())[0][8:10]
tmp2 = nn.ModuleList(AlexNet_Model.children())[0][10:]
tmp3 = nn.ModuleList(AlexNet_Model.children())[2][:]


# add redundant maxpool and remove avgpool layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


model = nn.Sequential(*tmp0, nn.MaxPool2d(kernel_size=1),
                      *tmp1, nn.MaxPool2d(kernel_size=1), *tmp2, Flatten(), *tmp3)

# move the input and model to GPU for speed if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Now using device:", device)

# basic setting of model
model_path = '../parameter/alex_normal_mnist.pth'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(AlexNet_Model.parameters(), lr=0.001, momentum=0.9)
epoches = 30

print("Start Training")

# Start training
best_acc = 0.0
for epoch in range(epoches):  # loop over the dataset multiple times
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    model.train()

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)
        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()

        # get the index of the class with the highest probability
        train_pred = torch.argmax(output, 1)
        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(valloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            val_pred = torch.argmax(outputs, 1)
            # get the index of the class with the highest probability
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
            val_loss += batch_loss.item()

        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, 15, train_acc /
            len(train_ds), train_loss/len(trainloader), val_acc /
            len(val_ds), val_loss/len(valloader)
        ))

        # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print('saving model with acc {:.3f}'.format(best_acc/len(val_ds)))
print('Finished training of AlexNet, model save at CAPM/parameter/', model_path, sep = '')


print('Start testing')
print('Start testing FGSM attacked Images')
model_test = nn.Sequential(*tmp0, nn.MaxPool2d(kernel_size=1),
                      *tmp1, nn.MaxPool2d(kernel_size=1), *tmp2, Flatten(), *tmp3)

model_test.load_state_dict(torch.load(model_path))
model_test.eval()

epsilon = (0.01)
epsilon = torch.tensor(epsilon).view(1, 1, 1).to(device)

print('starting test standard acc')
test_loss, test_acc = evaluate_standard(testloader, model_test, )
print('acc: ', 100*test_acc, '%', sep = '')
print('starting fgsm acc')
fgsm_loss, fgsm_acc = evaluate_fgsm(testloader, model_test, 50, 10, epsilon)
print('acc: ', 100*fgsm_acc, '%', sep = '')
print('starting pgd acc')
pgd_loss, pgd_acc = evaluate_pgd(testloader, model_test, 20, 10, epsilon)
print('acc: ', 100*pgd_acc, '%', sep = '')