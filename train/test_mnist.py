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

# model
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

# test data
# use same dataset (first 100 of Cifar10 test)
df = pd.read_csv('../data/mnist_test.csv')
data = df.iloc[:100, 1:].values.astype(float)
label = df.iloc[:100, 0].values

data_tensor = torch.from_numpy(data).float().view(
    len(data), -1, 28, 28)  # -------------------------------------------
target_tensor = torch.from_numpy(label).long()


transform_test = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,)) #--------------------------------------------
])

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
testloader = DataLoader(
    test_ds, shuffle=False, batch_size=4, num_workers=2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 

model_path = '../parameter/alex_normal_mnist.pth'

print('Start testing')
print('Start testing FGSM attacked Images')
model_test = nn.Sequential(*tmp0, nn.MaxPool2d(kernel_size=1),
                      *tmp1, nn.MaxPool2d(kernel_size=1), *tmp2, Flatten(), *tmp3)

model_test.load_state_dict(torch.load(model_path))
model_test.eval()

Eps = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.2, 0.3]

for epsilon in Eps:
    print('==============================================')
    print('epsilon =', epsilon)
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
    print('==============================================')
