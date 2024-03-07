#from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from os.path import join as oj
import torch.utils.data as utils
from torchvision import datasets, transforms
import numpy as np
from model import Net
import os
import sys
import pickle as pkl
from copy import deepcopy
from params_save import S # class to save objects
sys.path.append('../../src')
import score_funcs
from score_funcs import gradient_sum,eg_scores_2d,cdep
import cd
import random
model_path = "../../models/ColorMNIST_test"
import os
os.makedirs(model_path, exist_ok= True)
torch.backends.cudnn.deterministic = True #this makes results reproducible. 
def save(p,  out_name):
    # save final
    os.makedirs(model_path, exist_ok=True)
    pkl.dump(s._dict(), open(os.path.join(model_path, out_name + '.pkl'), 'wb'))
    
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

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--regularizer_rate', type=float, default=0, metavar='N',
                    help='how heavy to regularize lower order interaction (AKA color)')
parser.add_argument('--grad_method', type=int, default=0, metavar='N',
                    help='which gradient method is used - Grad or CD')
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()
s = S(args.epochs)
use_cuda = not args.no_cuda and torch.cuda.is_available()
regularizer_rate = args.regularizer_rate
s.regularizer_rate = regularizer_rate
num_blobs = 8
s.num_blobs = num_blobs
s.seed = args.seed


device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 0, 'pin_memory': True,  'worker_init_fn':np.random.seed(12)} if use_cuda else {}

x_numpy_train = np.load(oj("../../data/ColorMNIST",   "clarc_setting_train_x.npy"))
prob = (x_numpy_train.sum(axis = 1) > 0.0).mean(axis = 0).reshape(-1)
prob /=prob.sum()
mean = x_numpy_train.mean(axis = (0,2,3))
std = x_numpy_train.std(axis = (0,2,3)) 
#x_numpy /= std[None, :, None, None,]
#x_numpy -= mean[None, :, None, None,]
def load_dataset(name):
    x_numpy = np.load(oj("../../data/ColorMNIST", name + "_x.npy"))
    x_numpy -= mean[None, :, None, None,]
    x_numpy /= std[None, :, None, None,]
    y_numpy = np.load(oj("../../data/ColorMNIST", name +"_y.npy"))
    x_tensor = torch.Tensor(x_numpy)
    y_tensor = torch.Tensor(y_numpy).type(torch.int64)
    dataset = utils.TensorDataset(x_tensor,y_tensor) 

    return dataset
    

train_dataset = load_dataset("clarc_setting_train")
val_dataset = load_dataset("clarc_setting_val")
test_dataset = load_dataset("test")

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
train_loader = utils.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = utils.DataLoader(val_dataset,
    batch_size=args.test_batch_size, shuffle=True, **kwargs) 
test_loader = utils.DataLoader(test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs) 




blobs = np.zeros((28*28,28,28))
for i in range(28):
    for j in range(28):
        blobs[i*28+j, i, j] =1



model = MNISTColorNet2().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# optimizer = optim.Adam(model.parameters(), weight_decay = 0.001)

def train(args, model, device, train_loader, optimizer, epoch, regularizer_rate, until_batch = -1):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if until_batch !=-1 and batch_idx > until_batch:
            break
        data, target = data.to(device), target.to(device)
         
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        # if regularizer_rate !=0:
        #     print("here")
        #     add_loss = torch.zeros(1,).cuda()
        #     blob_idxs = np.random.choice(28*28, size = num_blobs, p = prob)
        #     if args.grad_method ==0:
        #         for i in range(num_blobs): 
        #             add_loss += score_funcs.cdep(model, data, blobs[blob_idxs[i]],model_type = 'mnist')
        #         (regularizer_rate*add_loss+loss).backward()
        #     elif args.grad_method ==1:
        #         for i in range(num_blobs): 
        #             add_loss +=score_funcs.gradient_sum(data, target, torch.FloatTensor(blobs[blob_idxs[i]]).to(device),  model, F.nll_loss)
        #         (regularizer_rate*add_loss).backward()
        #         loss = F.nll_loss(output, target)
        #         loss.backward()
        #     elif args.grad_method ==2:
        #         for j in range(len(data)):
        #             for i in range(num_blobs): 
        #                 add_loss +=(score_funcs.eg_scores_2d(model, data, j, target, 50) * torch.FloatTensor(blobs[blob_idxs[i]]).to(device)).sum()

        #         (regularizer_rate*add_loss).backward()
        #         loss = F.nll_loss(output, target)
        #         loss.backward()
        # else:
        add_loss =torch.zeros(1,)
        loss.backward()
         
        optimizer.step()
        
        
        if batch_idx % args.log_interval == 0:
            pred = output.argmax(dim=1, keepdim=True)
            acc = 100.*pred.eq(target.view_as(pred)).sum().item()/len(target)
            s.losses_train.append(loss.item())
            s.accs_train.append(acc)
            s.cd.append(add_loss.item())
   


def test(model, device, dataset_loader, is_test = False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in dataset_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataset_loader.dataset)

    
    if is_test:
        s.acc_test = 100. * correct / len(dataset_loader.dataset)
        s.loss_test = test_loss
        print('\Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataset_loader.dataset),
        100. * correct / len(dataset_loader.dataset)))
    else:
        
        s.losses_dev.append(test_loss)
        s.accs_dev.append(100. * correct / len(dataset_loader.dataset))
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataset_loader.dataset),
        100. * correct / len(dataset_loader.dataset)))
    return test_loss


 
if args.mode == 'train':
    best_model_weights = None
    best_test_loss = 100000
    patience = 0
    cur_patience = 0

    for epoch in range(1, args.epochs + 1):

        train(args, model, device, train_loader, optimizer, epoch, regularizer_rate)
        test_loss = test(model, device, val_loader)
        if test_loss < best_test_loss:
            
            cur_patience = 0
            best_test_loss = test_loss
            best_model_weights = deepcopy(model.state_dict())
        else:
            cur_patience +=1
            if cur_patience > patience:
                break
    model.load_state_dict(best_model_weights)

    test_loss = test(model, device, test_loader,is_test=True)
    print("test_loss",test_loss)
    torch.save(model.state_dict(),"clarc_setting_model_colorMNIST.pt")

if args.mode == 'test':
    name = 'clarc_setting_model_colorMNIST.pt'
    model.load_state_dict(torch.load('/home/avani.gupta/tcav_pt/'+name))
    print("loaded model from ",name)
    test_loss = test(model, device, val_loader)
    print("test_loss",test_loss)
# torch.save(net.state_dict(), model_type+'.pt')
# s.dataset= "Color"      
# test(args, model, device, test_loader, is_test = True)
# if args.grad_method ==0:
#     s.method = "CDEP"
# elif args.grad_method ==2:
#     s.method = "EGradients"
# else:
#     s.method = "Grad"
# #s.model_weights = best_model_weights
# np.random.seed()
# pid = ''.join(["%s" % np.random.randint(0, 9) for num in range(0, 20)])
# save(s,  pid)
