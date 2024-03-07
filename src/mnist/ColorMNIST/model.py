import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from os.path import join as oj
import torch.utils.data as utils
from torchvision import datasets, transforms
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    def logits(self, x):
    
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x



class MNISTColorNet2(nn.Module):
    def __init__(self):
        super(MNISTColorNet2, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
    def forward(self, x, z=None, v=None, bottleneck_name=None,use_aclarc=False):
        breakpoint()
        x = F.relu(self.conv1(x))
        x = self.dropout1(F.max_pool2d(x, 2, 2))
        x = self.dropout2(F.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        # if use_aclarc and bottleneck_name=='fc2':
        #     v_vt = v@v.T
        #     e = torch.eye(len(v_vt),device=v.device) - v_vt
        #     x = (e@x)@model.fc2.weight.T + v_vt@z b
        # else:
        x = self.fc1(x)
        x = F.relu(x)
        #we change input based on cav for next layer
        if use_aclarc and bottleneck_name=='fc1':
            v_vt = v@v.T
            e = torch.eye(len(v_vt),device=v.device) - v_vt
            x = e@x + v_vt@z
        
        x = self.fc2(x)
        if use_aclarc and bottleneck_name=='fc2':
            v_vt = v@v.T
            e = torch.eye(len(v_vt),device=v.device) - v_vt
            x = e@x + v_vt@z
        return F.log_softmax(x, dim=1)
        

    def logits(self, x):
    
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return x