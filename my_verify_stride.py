import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as f
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.autograd import Variable
import torch.utils.data as dset
import torch
import random
import time
import gc
import psutil
from PIL import Image
import pandas as pd
import torch.utils.data as Data
import argparse
import importlib
import os

preprocess_mnist = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,)) #--------------------------------------------
])

preprocess_cifar = transforms.Compose([
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ------------------------
])

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='smallNet', help='smallNet/variant_smallNet/largeNet/variant_largeNet or path to net file', type=str)
    parser.add_argument('--custom_class_name', default='', help='class name of the custom model', type=str)
    parser.add_argument('--pth_file', default='parameter/mnist_maxpool_best.pth', help='path to stored model', type=str)
    parser.add_argument('--epsilon', default=0.0, help='epsilon value of attack', type=float)
    parser.add_argument('--data', default='mnist', choices=['mnist', 'cifar10', 'svhn'], help='mnist/cifar10/svhn', type=str)
    parser.add_argument('-d', '--debug',  action = 'store_true')
    return parser.parse_args()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),-1)

# small Model
class smallNet(nn.Module):
    def __init__(self):
        super(smallNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),            
                
            Flatten(),
            
            nn.Linear(784,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, input):
        return self.main(input)

# variant small Model (can modify the stride in maxpool)
class variant_smallNet(nn.Module):
    def __init__(self):
        super(variant_smallNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=3),
                
            Flatten(),
            
            nn.Linear(400,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, input):
        return self.main(input)

# large Model
class largeNet(nn.Module):
    def __init__(self):
        super(largeNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              
                
            Flatten(),
            
            nn.Linear(576,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, input):
        return self.main(input)

# variant large Model
class variant_largeNet(nn.Module):
    def __init__(self):
        super(variant_largeNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
             
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=4),

            Flatten(),
            
            nn.Linear(576,100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, input):
        return self.main(input)

class convSmall(nn.Module):
    def __init__(self):
        super(convSmall, self).__init__()
        self.main = nn.Sequential(
            # ConvSmall (MNIST our)
            nn.Conv2d(1, 16, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),

            nn.Conv2d(16, 32, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),

            Flatten(),

            nn.Linear(800, 100), 
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.main(x)
        return x

class convMed(nn.Module):
    def __init__(self):
        super(convMed, self).__init__()
        self.main = nn.Sequential(
            # ConvMed (MNIST our)
            nn.Conv2d(1, 16, 2, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 2, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            Flatten(),

            nn.Linear(1568, 100), 
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.main(x)
        return x


class convSmallCIFAR10(nn.Module):
    def __init__(self):
        super(convSmallCIFAR10, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),

            nn.Conv2d(16, 32, 4, stride = 2, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),

            Flatten(),

            nn.Linear(1152, 100), 
            nn.ReLU(),
            nn.Linear(100, 10)
        )
    def forward(self, x):
        x = self.main(x)
        return x

class convSuperCIFAR10(nn.Module):
    def __init__(self):
        super(convSuperCIFAR10, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),

            nn.Conv2d(32, 32, 4, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),

            nn.Conv2d(32, 64, 3, stride = 1, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),

            nn.Conv2d(64, 64, 4, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride = 1),

            Flatten(),

            nn.Linear(30976, 512), 
            nn.ReLU(),

            nn.Linear(512, 512), 
            nn.ReLU(),

            nn.Linear(512, 10)
        )
    def forward(self, x):
        x = self.main(x)
        return x



args = get_args()

N = 0

if args.data == 'mnist':
    N = 28
    file_path = 'data/mnist_test.csv'
    preprocess = preprocess_mnist
elif args.data == 'cifar10':
    N = 32
    file_path = 'data/cifar10_test.csv'
    preprocess = preprocess_cifar
elif args.data == 'svhn':
    print('XXX: not done yet')
    pass
    


if args.net == 'smallNet':
    net = smallNet()
    pth_file = 'parameter/mnist_maxpool_best.pth'
elif args.net == 'variant_smallNet':
    net = variant_smallNet()
    pth_file = 'parameter/mnist_stride_best.pth'
elif args.net == 'largeNet':
    net = largeNet()
    pth_file = 'parameter/cifar10_maxpool_best.pth'
elif args.net == 'variant_largeNet':
    net = variant_largeNet()
    pth_file = 'parameter/cifar10_stride_best.pth'
elif args.net == 'convSmall':
    net = convSmall()
    pth_file = args.pth_file
elif args.net == 'convMed':
    net = convMed()
    pth_file = args.pth_file
elif args.net == 'convSmallCIFAR10':
    net = convSmallCIFAR10()
    pth_file = args.pth_file
# else:  # custom net
#     mo = importlib.import_module(args.net.replace('.py', ''))
#     mo = getattr(mo, args.custom_class_name)
#     net = mo()
#     pth_file = args.pth_file

print('file path:',file_path)
print('N:',N)
print('preprocess:',preprocess)

df = pd.read_csv(file_path)   #--------------------------------------------------------------

# get as numpy form
data = df.iloc[:100,1:].values.astype(float)
label = df.iloc[:100,0].values

# get as tensor form
data_tensor = torch.from_numpy(data).float().view(len(data),-1,N,N) #-------------------------------------------
target_tensor = torch.from_numpy(label).long()

class testset(Dataset):
    def __init__(self, data_tensor, target_tensor, loader = preprocess):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.loader = loader
        
    def __getitem__(self, index):
        data = self.data_tensor[index] / 255
        data = self.loader(data)
        target = self.target_tensor[index]
        return data,target
        
    def __len__(self):
        return self.data_tensor.size()[0]

# test load 
TestSet = testset(data_tensor, target_tensor)
testLoader = Data.DataLoader(TestSet, batch_size=len(TestSet), shuffle=False)
dataiter = iter(testLoader)
images, labels = dataiter.next()
print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)
print('len:',len(TestSet))


# print(images[0][0][0][:10])


print('pth_file:',pth_file)
# initialize the model ----------------------------------------------------------------
state = torch.load(pth_file,map_location='cpu')
net.load_state_dict(state)

print(net)



#f = open('_stride_'+str(0.01)+'.txt','w')
# f.write('total time : %f'%(total_end-total_start)+'  sec \n')
# f.write('ave time each img: %f'%(count_time/length)+'\n')
# f.write('verify acc: %f'%(acc/length))
# f.close()

################             prepare loading              #####################################



def get_lR_uR(zl,zu):
    
    uR = (zu > 0).detach().type_as(zu) * zu
    lR = (zl > 0).detach().type_as(zl) * zl
    
    return uR, lR

def get_matrix_d(zl,zu):
    
    d = (zl >= 0).detach().type_as(zl) # d check each point in I+ set
    I = ((zu > 0).detach() * (zl < 0).detach()) # I check each point in I set

    if I.sum().item() > 0:
            d[I] += zu[I]/(zu[I] - zl[I])
    
    d = d.t() # create the D matrix(in paper)
    
    return d

def premute_pixels(lR, uR, kM, sM):
    
    kh, kw = kM, kM  # kernel size
    dh, dw = sM, sM  # stride
    
    lR_patches = lR.unfold(2, kh, dh).unfold(3, kw, dw)
    uR_patches = uR.unfold(2, kh, dh).unfold(3, kw, dw)
    
    unfold_shape = lR_patches.size()
    
    lR_patches = lR_patches.contiguous().view(lR_patches.size(0), -1, kh*kw)
    uR_patches = uR_patches.contiguous().view(uR_patches.size(0), -1, kh*kw)
    
    return uR_patches, lR_patches, unfold_shape

LM_bds = []
UM_bds = []

def get_bar_u_l(lR, uR, reshape_size, kernel_size_M, stride_M):
    
    lR = lR.contiguous().view(*reshape_size)
    uR = uR.contiguous().view(*reshape_size)
    
    # permute the pixels which using for max-pool
    uR, lR, unfold_shape = premute_pixels(lR, uR, kernel_size_M, stride_M)
    
    n = lR.size(1)
    lM = torch.zeros(1,n)
    uM = torch.zeros(1,n)
    
    L_bar_bds = []
    U_bar_bds = []
    
    L_bds = []
    U_bds = []
    
    
    pixels_len = kernel_size_M * kernel_size_M # 每次kernel經過的pixel數
    
    print('input shape:', uR.size())
    print('uR:\n',uR[:,:5,:])
    print('lR:\n',lR[:,:5,:])

    for idx in range(pixels_len):
        l_bar = lR[:,:,idx] - uM
        u_bar = uR[:,:,idx] - lM
        
        # print('each u_bar shape:', u_bar.size())
        # print('each u_bar:',u_bar[:,:5])

        L_bar_bds.append(l_bar)
        U_bar_bds.append(u_bar)
        
        # get l', u'
        l_ = (l_bar > 0).detach().type_as(l_bar) * l_bar
        u_ = (u_bar > 0).detach().type_as(u_bar) * u_bar
        
        L_bds.append(l_)
        U_bds.append(u_)
        
        # get new lM, uM
        lM = lM + l_
        uM = uM + u_
    
    L_bar = torch.stack(L_bar_bds, dim = 0)
    U_bar = torch.stack(U_bar_bds, dim = 0)
    
    L_ = torch.stack(L_bds, dim = 0) 
    U_ = torch.stack(U_bds, dim = 0) 
    
    ### use for test maxpool layer's upper/lower bound
    LM_bds.append(lM)
    UM_bds.append(uM)

    # print('output shape:',U_bar.shape)
    # print('part of output:\n',U_bar[:,:,:5])
    # print('\n')

    print('original U_bar size:',U_bar.size())
    print('original U_bar size:\n',U_bar[:,:,:5])
    print('original L_bar size:\n',L_bar[:,:,:5])

    return U_bar, L_bar, L_, U_

def limit_get_bar_u_l_(lR, uR, reshape_size, kernel_size_M, stride_M):

    lR = lR.contiguous().view(*reshape_size)
    uR = uR.contiguous().view(*reshape_size)
    
    # permute the pixels which using for max-pool
    uR, lR, unfold_shape = premute_pixels(lR, uR, kernel_size_M, stride_M)
    
    n = lR.size(1)
    lM = torch.zeros(1,n)
    uM = torch.zeros(1,n)
    
    L_bar_bds = []
    U_bar_bds = []    

    #print('origin lR size:', lR[:,100:110,])
    
    pixels_len = kernel_size_M * kernel_size_M # 每次kernel經過的pixel數

    #print('maximun way')
    for idx in range(pixels_len):

        # step 1
        l_bar = lR[:,:,idx] - uM
        u_bar = uR[:,:,idx] - lM

        L_bar_bds.append(l_bar)
        U_bar_bds.append(u_bar)

        #print('idx:',idx, ' ', lR[:,:,idx].size())

        m = nn.MaxPool1d(idx+1, stride=pixels_len)
        lM = m(lR).squeeze(0).t()
        uM = m(uR).squeeze(0).t()

        # if idx < 3:
        #     print(idx+1,' uM:',uM[:,100:110])

    
    L_bar = torch.stack(L_bar_bds, dim = 0)
    U_bar = torch.stack(U_bar_bds, dim = 0)
    
    # print('iter U_bar:',U_bar[:,:,2690:])
    # print('iter L_bar:',L_bar[:,:,2690:])

    return U_bar, L_bar

def iter_get_bar_u_l_(lR, uR, reshape_size, kernel_size_M, stride_M):


    lR = lR.contiguous().view(*reshape_size)
    uR = uR.contiguous().view(*reshape_size)
    
    # permute the pixels which using for max-pool
    uR, lR, unfold_shape = premute_pixels(lR, uR, kernel_size_M, stride_M)
    
    n = lR.size(1)
    lM = torch.zeros(1,n)
    uM = torch.zeros(1,n)
    
    L_bar_bds = []
    U_bar_bds = []    
    
    pixels_len = kernel_size_M * kernel_size_M # 每次kernel經過的pixel數

    #print('uR:\n', uR[:,100:110,])

    #print('iterative way')
    for idx in range(pixels_len):
        
        # print('idx:',idx)
        # print('uM;\n',uM[:,2690:])
        # print('lM;\n',uM[:,2690:])

        # step 1
        l_bar = lR[:,:,idx] - uM
        u_bar = uR[:,:,idx] - lM

        L_bar_bds.append(l_bar)
        U_bar_bds.append(u_bar)

        
        # step 2
        I_u0 = (u_bar <= 0).detach().float()
        I_l0 = (l_bar < 0).detach().float()
        I_l0u = ((l_bar < 0).detach() * (u_bar > 0).detach()).float()
        I_0l = (l_bar >= 0).detach().float()
        
        a = u_bar/(u_bar - l_bar)
        b = - (l_bar*u_bar)/(u_bar - l_bar)
        a[a != a] = 0
        b[b != b] = 0

        check_inf_a = torch.isinf(a)
        check_inf_b = torch.isinf(b)
        a[check_inf_a] = 0
        b[check_inf_b] = 0

        uM = uM*I_l0 + uR[:,:,idx]*I_0l
        lM = lM*I_u0 + ( (1-a)*lM + a*lR[:,:,idx] + b )*I_l0u + lR[:,:,idx]*I_0l

        # if idx < 3:
        #     print(idx+1, ' uM:',uM[:,100:110])

    
    L_bar = torch.stack(L_bar_bds, dim = 0)
    U_bar = torch.stack(U_bar_bds, dim = 0)
    
    # print('iter U_bar:',U_bar[:,:,2690:])
    # print('iter L_bar:',L_bar[:,:,2690:])

    return U_bar, L_bar


def get_uM_lM(new_uR, new_lR):
    
    u_bar = (new_uR[:,0] - new_lR[:,1])
    l_bar = (new_lR[:,0] - new_uR[:,1])

    m = nn.ReLU()

    uM = (m(u_bar) + new_uR[:,1]).unsqueeze(0)
    lM = (m(l_bar) + new_lR[:,1]).unsqueeze(0)

    return uM, lM

def tighter_get_bar_u_l(lR, uR, reshape_size, kernel_size_M, stride_M):
    
    lR = lR.contiguous().view(*reshape_size)
    uR = uR.contiguous().view(*reshape_size)
    
    # permute the pixels which using for max-pool
    uR, lR, unfold_shape = premute_pixels(lR, uR, kernel_size_M, stride_M)
    
    n = lR.size(1)
    lM = torch.zeros(1,n)
    uM = torch.zeros(1,n)
    
    L_bar_bds = []
    U_bar_bds = []
    
    L_bds = []
    U_bds = []
    
    print('input shape:', uR.size())
    # print('uR:\n',uR[:,:5,:])
    # print('lR:\n',lR[:,:5,:])

    batch_size = uR.size(0)
    blocks = uR.size(1)
    compare_compoments = uR.size(2)
    

    for idx in range(2): # <---------------- range update to 2
        l_bar = lR[:,:,idx] - uM
        u_bar = uR[:,:,idx] - lM
        
        # print('each u_bar shape:', u_bar.size())
        # print('each u_bar:',u_bar[0][:5])

        L_bar_bds.append(l_bar)
        U_bar_bds.append(u_bar)
        
        # get l', u'
        l_ = (l_bar > 0).detach().type_as(l_bar) * l_bar
        u_ = (u_bar > 0).detach().type_as(u_bar) * u_bar
        
        L_bds.append(l_)
        U_bds.append(u_)
        
        # get new lM, uM
        lM = lM + l_
        uM = uM + u_
    
    # <---------------------------------- add another for here. Note we should append the new ubar & lbar

    for idx in range(2,compare_compoments):
    
        max_idx = torch.max(uR[:,:,:idx], 2).indices
        min_idx = torch.min(lR[:,:,:idx], 2).indices

        # print(max_idx)
        # print(min_idx)
        
        uR_max_tensor = torch.stack([uR[:,i,j] for i,j in zip(torch.arange(0, blocks), max_idx[0])])
        uR_min_tensor = torch.stack([uR[:,i,j] for i,j in zip(torch.arange(0, blocks), min_idx[0])])

        lR_max_tensor = torch.stack([lR[:,i,j] for i,j in zip(torch.arange(0, blocks), max_idx[0])])
        lR_min_tensor = torch.stack([lR[:,i,j] for i,j in zip(torch.arange(0, blocks), min_idx[0])])

        new_uR = torch.cat((uR_max_tensor,uR_min_tensor),1)
        new_lR = torch.cat((lR_max_tensor,lR_min_tensor),1)
        
        new_uM, new_lM = get_uM_lM(new_uR, new_lR)
        u_bar = uR[:,:,idx] - new_lM
        l_bar = lR[:,:,idx] - new_uM
        
        L_bar_bds.append(l_bar)
        U_bar_bds.append(u_bar)
    
    L_bar = torch.stack(L_bar_bds, dim = 0)
    U_bar = torch.stack(U_bar_bds, dim = 0)
    
    L_ = torch.stack(L_bds, dim = 0) 
    U_ = torch.stack(U_bds, dim = 0) 

    print('tighter U_bar size:\n',U_bar[:,:,:5])
    print('tighter L_bar size:\n',L_bar[:,:,:5])
    
    return U_bar, L_bar, L_, U_

new_LM = []
new_UM = []

def new_get_bar_u_l(lR, uR, reshape_size, kernel_size_M, stride_M):
    
    new_LM_bds = []
    new_UM_bds = []
    
    lR = lR.contiguous().view(*reshape_size)
    uR = uR.contiguous().view(*reshape_size)
    
    # permute the pixels which using for max-pool
    uR, lR, unfold_shape = premute_pixels(lR, uR, kernel_size_M, stride_M)
    
    UR = uR.permute(0,2,1).squeeze(0)
    LR = lR.permute(0,2,1).squeeze(0)
    
    
    sample_num = 100000
    true_UBar = []
    true_LBar = []
    
    true_U_ = []
    true_L_ = []

    for col in range(LR.size(1)):
        #print(col)
        zM = torch.zeros(sample_num)

        for row in range(LR.size(0)):
            #print(LR[row][col], UR[row][col])

            zR = torch.Tensor(sample_num).uniform_(LR[row][col],UR[row][col])
            zBar = zR - zM

            true_UBar.append( torch.max(zBar) )
            true_LBar.append( torch.min(zBar) )
            #print(true_UBar[-1],true_LBar[-1])

            z_ = torch.max(zBar,torch.zeros(sample_num))
            true_U_.append(torch.max(z_))
            true_L_.append(torch.min(z_))
            
            zM = zM + z_


        new_LM_bds.append(torch.min(zM))
        new_UM_bds.append(torch.max(zM))

    U_Bar = torch.stack(true_UBar, dim = 0).reshape(LR.size(1),LR.size(0)).permute(1,0)
    L_Bar = torch.stack(true_LBar, dim = 0).reshape(LR.size(1),LR.size(0)).permute(1,0)
    
    U_Bar = U_Bar.unsqueeze(1)
    L_Bar = L_Bar.unsqueeze(1)
    
    U_ = torch.stack(true_U_, dim = 0).reshape(LR.size(1),LR.size(0)).permute(1,0)
    L_ = torch.stack(true_L_, dim = 0).reshape(LR.size(1),LR.size(0)).permute(1,0)
    
    U_ = U_.unsqueeze(1)
    L_ = L_.unsqueeze(1)
    
    new_LM.append( torch.stack(new_LM_bds, dim = 0).unsqueeze(0) )
    new_UM.append( torch.stack(new_UM_bds, dim = 0).unsqueeze(0)  )

    print('random U_bar size:\n',U_Bar[:,:,:5])
    print('random L_bar size:\n',L_Bar[:,:,:5])
    
    return U_Bar, L_Bar, L_, U_

def get_Kappa(Rho, d_bar, kM, sM, map_size_InMaxpool, map_size_OutMaxpool): #unfold_shape, pad_num):
    
    list_Kappa = []
    
    for idx in range(d_bar.size(0)-1,-1,-1):
        kappa = d_bar[idx]*Rho
        Rho -= kappa
        list_Kappa.append(kappa)
    
    #print('kappa list:\n',list_Kappa)
    Kappa = torch.stack(list_Kappa, dim = 1)
    #print('stack kappa:\n',Kappa)
    
    Kappa = torch.flip(Kappa,[1])
    kappa_hat = Kappa
    
    
    #print('kappa_hat:\n',kappa_hat, kappa_hat.shape)
    
    
    
    ###################################################### Fold add
    
    #kernel_size = kM      ########## should be got from the function input
    out_H, out_W = map_size_OutMaxpool[2], map_size_OutMaxpool[3]
    flat_kernelLength = out_H*out_W
    stride = flat_kernelLength
    In_H, In_W = map_size_InMaxpool[2], map_size_InMaxpool[3]
    
    #print('k, s, In_H, In_W:',kM, sM, In_H, In_W)
    #print('out_H, out_W:', out_H, out_W)
    
    
    temp = kappa_hat.permute(0,2,1,3).unfold(2, kM*kM, flat_kernelLength).unfold(3, flat_kernelLength, stride)
    #print('temp:',temp, temp.shape)
    input_kappa = temp.reshape(kappa_hat.size(0),-1, flat_kernelLength)
    #print('input_kappa:\n', input_kappa, input_kappa.shape)
    
    fold = torch.nn.Fold(output_size=(In_H, In_W), kernel_size=(kM, kM), stride = sM) ########## should be got from the function input
    
    
    test_kappa_orig = fold(input_kappa)
    Kappa_orig = test_kappa_orig
    #print('test_kappa_orig:\n',test_kappa_orig, test_kappa_orig.shape)
    

    
    del list_Kappa 
    del kappa 
    gc.collect()
    
    return Kappa_orig, kappa_hat

# backward 
    
def get_upper_lower(x, epsilon, model, out_layer_size, 
                    D = None, Lower = None, Upper = None, 
                    D_bar = None, bar_Lower = None, bar_Upper = None,
                    check_evaluate = False, y = None):
    
    nu = []
    
    ### set nu:
    if check_evaluate == False : 
        if isinstance(model[-1], nn.Linear):
            n = out_layer_size[-1][1] # 計算最後一層layer的output neuro數 
            nu.append(torch.eye(n))#nu_4

        if isinstance(model[-1], nn.Conv2d):
            n = out_layer_size[-1][1]*out_layer_size[-1][2]*out_layer_size[-1][3]
            nu.append(torch.eye(n).view(n,*out_layer_size[-1][1:]))
    
    elif check_evaluate == True :
            n = out_layer_size[-1][1] 
            nu.append((torch.eye(n) - torch.eye(n)[y]).t()) # c = y_targ - y_true
    
    nu_b = []
    nu_x = [x]
    l1 = []
    Big_Kappa = []
    
    upper_max_pool_term = []
    lower_max_pool_term = []
    
    upper_relu_term = []
    lower_relu_term = []
    

    model_len = len(out_layer_size)
    
    #print('All out layer size:',out_layer_size)
    
    
    if D != None :
        pos_Upper = len(Upper) - 1
        pos_Lower = len(Lower) - 1
        pos_bar_U = len(bar_Upper) - 1
        pos_bar_L = len(bar_Lower) - 1
        pos_d = len(D) - 1
        pos_bar_d = len(D_bar) -1
    
    for idx, layer in enumerate(reversed(model)):
        
        if isinstance(layer, nn.Linear):
            #print('Linear:',(psutil.virtual_memory().used)/(10**9))
            W = layer.weight
            b = layer.bias.unsqueeze(0)

            nu_b.append(torch.mm(b,nu[-1]).detach())
            nu.append(torch.mm(W.t(),nu[-1]).detach())
        
        if isinstance(layer, Flatten):
            # here we reshape the nu[-1]
            pos = (model_len - 1) - (idx + 1)
            re_size = out_layer_size[pos] 
            # idx + 1 : next layer ; model_len - 1 : the last num of listout_layer_size() 

            nu.append(nu[-1].t().view(n,*re_size[1:]))
            
        if isinstance(layer, nn.Conv2d):
            
            ### declare the nus
            pos = (model_len - 1) - idx
            flat_num = out_layer_size[pos][1]*out_layer_size[pos][2]*out_layer_size[pos][3]


            # accroding to this layer's output picture size (output means the forward result via this layer)
            # 每張圖被 flat 成 vector 後 ，vector中所含的 neuro數，為每張圖的 total neuros = W*H*Chl
            nus = nu[-1].flatten().view(n,flat_num).t() 

            ### get the weight and bias

            in_chl = layer.out_channels
            out_chl = layer.in_channels
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            pad = layer.padding[0]

            in_size = out_layer_size[pos-1][2]
            out_size = out_layer_size[pos][2]

            out_pad = in_size - (out_size -1)* stride + 2*pad - kernel_size

            dc = nn.ConvTranspose2d(in_chl,out_chl,kernel_size,stride,pad,bias = False,output_padding = out_pad)
            dc.weight = layer.weight
            b = layer.bias
            b = b.view(b.size()[0],1).expand(b.size()[0],out_layer_size[pos][2]*out_layer_size[pos][3])
            b = b.flatten().unsqueeze(0)
            ### get nu and nu_b

            nu_b.append(torch.mm(b,nus))
            nu.append(dc(nu[-1]))
            
            #print('conv:',(psutil.virtual_memory().used)/(10**9))
        
        if isinstance(layer, nn.MaxPool2d):
            
            
            pos = (model_len - 1) - idx - 1 
            temp = torch.rand(*(out_layer_size[pos])) # out_layer_size[pos] is size after ReLU layer (f_l) 
            
            #print('maxpool out_layer_size:',out_layer_size[pos])
            kM = layer.kernel_size
            sM = layer.stride
            map_size_InMaxpool = out_layer_size[pos]
            map_size_OutMaxpool = out_layer_size[pos+1]
            
            # get kappa
            Beta = nu[-1].view(n,1,-1)
            Rho = Beta
            
            kh, kw = kM, kM  # kernel size
            dh, dw = kM, kM  # stride
            
            # patches = temp.unfold(2, kh, dh).unfold(3, kw, dw)
            # unfold_shape = patches.size()
    
            # # prepare for padding size
            # N_l = out_layer_size[pos+1][2] # out_layer_size[pos] is size after max-pool layer (N_l) 
            # f_l = out_layer_size[pos][2]   # out_layer_size[pos] is size after ReLU layer (f_l) 
            # r_l = kM*N_l 
            # pad_num = f_l - r_l
            
            Kappa_orig, kappa_hat = get_Kappa(Rho, D_bar[pos_bar_d], kM, sM, map_size_InMaxpool, map_size_OutMaxpool)#, unfold_shape, pad_num)
            Big_Kappa.append(Kappa_orig)
            
            # count max_pool_term
            bar_I = ((bar_Upper[pos_bar_U] > 0).detach() * (bar_Lower[pos_bar_L] < 0).detach()).float()


            upper_max_pool_term.append( torch.sum( bar_Lower[pos_bar_L]*(bar_I*kappa_hat).clamp(0), (3,2,1) ).detach() )
            lower_max_pool_term.append( torch.sum( bar_Lower[pos_bar_L]*(-1*bar_I*kappa_hat).clamp(0), (3,2,1)).detach()) 
            

            pos_bar_U -= 1
            pos_bar_L -= 1
            pos_bar_d -= 1

            del kappa_hat
            del Kappa_orig
            gc.collect()
            
            #print('max-pool:',(psutil.virtual_memory().used)/(10**9))
        
        if isinstance(layer, nn.ReLU):
            
            
            I = ((Upper[pos_Upper] > 0) * (Lower[pos_Lower] < 0)).float() 
            
            if len(Big_Kappa) != 0 :
            
                flat_num = torch.numel(Big_Kappa[-1][0])
                nus = Big_Kappa[-1].flatten().view(n,flat_num)* D[pos_d] 
                origin_size = Big_Kappa[-1].size()
                nu.append( nus.view(*origin_size) )
                           
                upper_relu_term.append( (Lower[pos_Lower]*(I*nus).clamp(min = 0)).t().sum(0).detach())
                lower_relu_term.append( (Lower[pos_Lower]*(-1*I*nus).clamp(min = 0)).t().sum(0).detach() )
            
            else :
                nus = nu[-1] * (D[pos_d].t())
                nu.append(nus)

                upper_relu_term.append( (Lower[pos_Lower].t()*(I.t()*nus).clamp(min = 0)).sum(0).detach())
                lower_relu_term.append( (Lower[pos_Lower].t()*(-1*I.t()*nus).clamp(min = 0)).sum(0).detach() )
            
            pos_d -= 1
            pos_Lower -= 1
            pos_Upper -= 1
            
            #print('ReLU:',(psutil.virtual_memory().used)/(10**9))
            
    #####  get nu_x and l1  #####        

    #nu_1
    nu_x.append(torch.sum(nu[-1]*nu_x[-1],(3,2,1)))
    nu_x[-1] = nu_x[-1].view(1, nu_x[-1].size()[0])

    l1.append(torch.sum(nu[-1].abs(),(3,2,1)))
    l1.append(l1[-1].view(1,l1[-1].size()[0]))
            

        
    ##### get the upper lower bound #####
    if D == None :
            upper = sum(nu_b) + nu_x[-1] + epsilon*l1[-1]
            lower = sum(nu_b) + nu_x[-1] - epsilon*l1[-1]
        
    else:
        upper = sum(nu_b) + nu_x[-1] + epsilon*l1[-1] - sum(upper_relu_term) - sum(upper_max_pool_term)
        lower = sum(nu_b) + nu_x[-1] - epsilon*l1[-1] + sum(lower_relu_term) + sum(lower_max_pool_term)

    del nu
    del nu_b 
    del nu_x
    del l1
    del Big_Kappa

    
    del upper_max_pool_term
    del lower_max_pool_term
    
    del upper_relu_term
    del lower_relu_term
    gc.collect()
    
    return upper, lower

def collect_all_upper_lower(model, x, y, epsilon):
    
    in_pic = x
    out_layer_size = [in_pic.size()]
    net = []

    lower_bds = []
    upper_bds= []
    D = []
    D_bar = []

    uR_bds = []
    lR_bds = []

    bar_upper_bds = []
    bar_lower_bds = []

    UNFOLD_shape = []

    # get each layer's output size
    for idx,layer in enumerate(model):

        #print('\n\n',idx,layer)

        # save the pic size after each layer
        out_pic = layer(in_pic)
        out_layer_size.append(out_pic.size())
        in_pic = out_pic

        # stack the net
        net.append(layer)

        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            
            if idx == 0:
                upper, lower = get_upper_lower(x, epsilon, net, out_layer_size)
            else:
                upper, lower = get_upper_lower(x, epsilon, net, out_layer_size,
                                               D, lower_bds, upper_bds, 
                                               D_bar,bar_lower_bds, bar_upper_bds)


            lower_bds.append(lower.detach())
            upper_bds.append(upper.detach())

            
            del lower
            del upper
            gc.collect()
            
        if isinstance(layer, nn.ReLU):
            
            d = get_matrix_d(lower_bds[-1],upper_bds[-1]).t()
            D.append(d)

            uR, lR = get_lR_uR(lower_bds[-1], upper_bds[-1])

            uR_bds.append(uR)
            lR_bds.append(lR)
            
            #print('**end f RelU:',(psutil.virtual_memory().used)/(10**9))
            
        if isinstance(layer, nn.MaxPool2d):

            kM = layer.kernel_size
            sM = layer.stride
            # loose method find upper lower
            #u_bar, l_bar, _, _ = get_bar_u_l(lR_bds[-1], uR_bds[-1], out_layer_size[-2], kM, sM)
            #print('iter u_bar size:',u_bar.size(), u_bar.dtype)

            # tighter method find upper lower
            #u_bar, l_bar, _, _ = tighter_get_bar_u_l(lR_bds[-1], uR_bds[-1], out_layer_size[-2], kM, sM)

            #real points find upper lower
            #u_bar, l_bar, _, _ = new_get_bar_u_l(lR_bds[-1], uR_bds[-1], out_layer_size[-2], kM, sM)
            #print('real u_bar size:',u_bar.size())

            # iterative method for finding upper lower
            u_bar, l_bar = iter_get_bar_u_l_(lR_bds[-1], uR_bds[-1], out_layer_size[-2], kM, sM)

            # maximun way
            u_bar, l_bar = limit_get_bar_u_l_(lR_bds[-1], uR_bds[-1], out_layer_size[-2], kM, sM)
            
            #print('check u_bar nan:',torch.sum(torch.isnan(u_bar)))
            #print('check l_bar nan:',torch.sum(torch.isnan(l_bar)))

            bar_upper_bds.append(u_bar)
            bar_lower_bds.append(l_bar)

            u_ = u_bar.flatten().unsqueeze(0)
            l_ = l_bar.flatten().unsqueeze(0)

            #print('u_:',u_)
            #print('l_:',l_)

            d_bar = get_matrix_d(l_,u_).t()
            d_bar = d_bar.view(*(u_bar.size()))

            

            D_bar.append(d_bar)

            #print('d_bar:',D_bar[-1])
            #print('**end f Maxpool:',(psutil.virtual_memory().used)/(10**9))
            

    #print('\n')
    #print('lower:',lower_bds[-1])
    #print('upper:',upper_bds[-1])
    
    #evaluate
    objective, _ = get_upper_lower( x, epsilon, model, out_layer_size, 
                               D, lower_bds[:-1], upper_bds[:-1], 
                               D_bar,bar_lower_bds, bar_upper_bds,
                               True, y)
    
    #print('objective:',objective)
    return lower_bds, upper_bds, objective

length = len(images)
#length = 100
# 到時X換成 MNIST test image 
# X = torch.randint(1,10,(1,images.size(1),images.size(2),images.size(3))).float()
# y = 3

acc = 0
count_time = 0.0
total_start = time.time()
for i in range(length):

    X = images[i].unsqueeze(0)
    Y = labels[i]
    Epsilon = args.epsilon
    #print(X.view(784,-1)[200:300].view(10,10))

    #print('**start ----------:',(psutil.virtual_memory().used)/(10**9))
    # get all upper/ lower bounds
    start = time.time()
    Lower_bds, Upper_bds,  objective = collect_all_upper_lower(net.main,X,Y,Epsilon)
    end = time.time()

    acc += (objective.max(1, keepdim=True)[1] == Y).sum().item()
    
    
#     start = time.time()
#     test_y = check_corectness(test_num = 100000, epsilon = Epsilon, X = X, net = net.main, upper = Upper_bds[-1], lower = Lower_bds[-1])
#     end = time.time()


        
    #print('objective:',objective)
    count_time += (end-start)
    if args.debug:
        print(i)
        print('Y:',Y)
        print('acc:',acc)
        print('time:',end-start)
        print('\n')
        print('ave time each img:',count_time/length)
    Lower_bds = None
    Upper_bds = None
    objective = None
    gc.collect()
    #print(acc/length)
    #print('**end ----------:',(psutil.virtual_memory().used)/(10**9))

    # if i == 0:
    #     break

total_end = time.time()

data_name = args.net.replace('.py', '')
if not os.path.exists('stride_result'):
    os.makedirs('stride_result')

f = open('stride_result/'+data_name +'_stride_'+str(Epsilon)+'.txt','w+')
f.write('total time : %f'%(total_end-total_start)+'  sec \n')
f.write('ave time each img: %f'%(count_time/length)+'\n')
f.write('verify acc: %f\n'%(acc/length))
f.close()
print("===============================================================")
print('epsilon =', Epsilon)
print(pth_file)
print('ave time each img:', (count_time/length))
print('verify acc:', (acc/length))
print("===============================================================")
