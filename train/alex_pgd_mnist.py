import argparse
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex import amp
import torchvision
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.utils.data as Data
from torchvision import transforms
import pandas as pd
import torch.optim as optim

from utils_mnist import (upper_limit, lower_limit, std, clamp, get_loaders, get_loaders_2,
                   attack_pgd, evaluate_pgd, evaluate_standard)


logger = logging.getLogger(__name__)

# me
transform = transforms.Compose([
    # transforms.Resize((227, 227)),
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
df = pd.read_csv('./data/mnist_test.csv')
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
# AlexNet_Model.classifier[1] = nn.Linear(9216, 2048)
# AlexNet_Model.classifier[4] = nn.Linear(2048, 512)
# AlexNet_Model.classifier[6] = nn.Linear(512, 10)
tmp00 = nn.ModuleList(AlexNet_Model.children())[0][0:1]
tmp0 = nn.ModuleList(AlexNet_Model.children())[0][1:8]
tmp1 = nn.ModuleList(AlexNet_Model.children())[0][8:10]
tmp2 = nn.ModuleList(AlexNet_Model.children())[0][10:]
tmp3 = nn.ModuleList(AlexNet_Model.children())[2][:]

# AlexNet_Model.classifier[1] = nn.Linear(9216, 2048)
# AlexNet_Model.classifier[4] = nn.Linear(2048, 512)
# AlexNet_Model.classifier[6] = nn.Linear(512, 10)
tmp00 = nn.ModuleList(AlexNet_Model.children())[0][0:1]
tmp0 = nn.ModuleList(AlexNet_Model.children())[0][1:8]
tmp1 = nn.ModuleList(AlexNet_Model.children())[0][8:10]
tmp2 = nn.ModuleList(AlexNet_Model.children())[0][10:]
tmp3 = nn.ModuleList(AlexNet_Model.children())[2][:]


# add redundant maxpool and remove avgpool layer
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

model = nn.Sequential(*tmp00, *tmp0, nn.MaxPool2d(kernel_size=1),
                      *tmp1, nn.MaxPool2d(kernel_size=1), *tmp2, Flatten(), *tmp3)

# move the input and model to GPU for speed if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Now using device:", device)

# basic setting of model
model_path = './parameter/alex_pgd_mnist.pth'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(AlexNet_Model.parameters(), lr=0.001, momentum=0.9)
epoches = 30

print("Start Training")

# endme

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../../cifar-data', type=str)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=7, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='train_pgd_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


args = get_args()

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
logfile = os.path.join(args.out_dir, 'output.log')
if os.path.exists(logfile):
    os.remove(logfile)

logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
    filename=logfile)
logger.info(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
train_loader, test_loader = trainloader, testloader

epsilon = (args.epsilon / 255.) / std
alpha = (args.alpha / 255.) / std

model.train()

opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
if args.opt_level == 'O2':
    amp_args['master_weights'] = args.master_weights
model, opt = amp.initialize(model, opt, **amp_args)
criterion = nn.CrossEntropyLoss()

lr_steps = args.epochs * len(train_loader)
if args.lr_schedule == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
        step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
elif args.lr_schedule == 'multistep':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

# Training
start_train_time = time.time()
logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
for epoch in range(args.epochs):
    print('now at epoch', epoch)
    start_epoch_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    for i, (X, y) in enumerate(train_loader):
        X, y = X.cuda(), y.cuda()
        delta = torch.zeros_like(X).cuda()
        if args.delta_init == 'random':
            for i in range(len(epsilon)):
                delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for ttt in range(args.attack_iters):
            output = model(X + delta)
            loss = criterion(output, y)
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            grad = delta.grad.detach()
            delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
            delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.grad.zero_()
        delta = delta.detach()
        output = model(X + delta)
        loss = criterion(output, y)
        opt.zero_grad()
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
        opt.step()
        train_loss += loss.item() * y.size(0)
        train_acc += (output.max(1)[1] == y).sum().item()
        train_n += y.size(0)
        scheduler.step()
    epoch_time = time.time()
    lr = scheduler.get_lr()[0]
    logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
        epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
train_time = time.time()
torch.save(model.state_dict(), './parameter' + model_path)
logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)
