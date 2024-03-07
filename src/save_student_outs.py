from turtle import Shape
import torch
import torch.nn as nn
import numpy as np
from utils.utils_tcav2 import *
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pickle
from torchvision.models import resnet18, resnet50
import numpy as np
from six.moves import range
from sklearn import linear_model, svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from utils import tcav_utils  as utils
import tensorflow as tf
import yaml
import numpy as np
import PIL
import matplotlib.pyplot as plt
import cv2
import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import os
from os import listdir
from os.path import isfile, join
import cv2
from io import StringIO 
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import abc
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from CGIntrinsics.models import networks
from utils.networks_classi import *
from utils.utils_train import *
import random
import torch.utils.data as tutils
import argparse
from datetime import datetime
import json
from argparse import ArgumentParser
from utils.imdb_classi_test import *
import torch.nn.functional as F
import abc
import torchvision.models as models
from torch.autograd import Variable

def get_args():
    parser = argparse.ArgumentParser()
    # generic params
    parser.add_argument(
        "--name",
        default=datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        help="Name to store the log file as",)
    parser.add_argument(
        "--seed", type=int, default=10, help="Random generator seed for all frameworks")
    parser.add_argument("--resume", help="Path to log file to resume from")
    parser.add_argument("--mode", default="test", help="train, test, or plot")
    parser.add_argument("--data-dir", default="/media/Data2/avani.gupta/CLEVR-Hans3/", help="Directory to data")
    parser.add_argument("--fp-ckpt", type=str, default='/home/avani.gupta/tcav_pt/NeSyXIL/src/clevr_hans/cnn/runs/conf_3/resnet-clevr-hans-17-conf_3_seed0/model_epoch56_bestvalloss_0.0132.pth', help="checkpoint filepath")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train with")
    parser.add_argument("--lr", type=float, default=1e-2, help="Outer learning rate of model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size to train with")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of threads for data loader"
    )
    parser.add_argument(
        "--dataset",
        default = "clevr-hans-state",
        choices=["clevr-hans-state"],
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Run on CPU instead of GPU (not recommended)",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Only run training, no evaluation"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only run evaluation, no training"
    )

    args = parser.parse_args()

    assert args.data_dir.endswith(os.path.sep)
    args.conf_version = args.data_dir.split(os.path.sep)[-2]
    args.name = args.name + f"-{args.conf_version}"

    if args.mode == 'test':
        assert args.fp_ckpt

    if args.no_cuda:
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    # tutils.seed_everything(args.seed)

    return args



class ConvsNN(torch.nn.Module):
    def __init__(self):
        super(ConvsNN,self).__init__()
        # kernel_size, stride = calc_params(shape=(384,512))
        self.conv1 = nn.ConvTranspose2d(64,32,(5,3),stride=2)
        self.conv2 = nn.ConvTranspose2d(32,16,(10,6),stride=1)
        self.conv3 = nn.ConvTranspose2d(16,4,(16,2),stride=2)
        self.conv4 = nn.ConvTranspose2d(4,1,(155,245),stride=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

class ConvsdownNN(torch.nn.Module):
    def __init__(self):
        super(ConvsdownNN,self).__init__()
        # kernel_size, stride = calc_params(shape=(384,512))
        self.conv1 = nn.Conv2d(1,4,(155,245),stride=1)
        self.conv2 = nn.Conv2d(4,16,(16,2),stride=2)
        self.conv3 = nn.Conv2d(16,32,(10,6),stride=1)
        self.conv4 = nn.Conv2d(32,64,(5,3),stride=2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
    
class fchead(torch.nn.Module):
    def __init__(self,input_size=196608, num_classes=2):
        super(fchead,self).__init__()
        self.linear = torch.nn.Linear(input_size,num_classes)
    
    def forward(self,feature):
        feature = feature.view(-1,196608)
        output = torch.sigmoid(self.linear(feature))
        return output




class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x, latent_vec=False):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return

class ResNet_Enco(AbstractAutoEncoder):
    def __init__(
        self, fc_hidden1=1024,
            fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_Enco, self).__init__()
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
        self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        self.convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans11 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        return x

    def decode(self, x):
        x = self.convTrans10(x)
        x = self.convTrans11(x)
        x = self.convTrans12(x)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(224, 224), mode='bilinear')
        return x
    def forward(self, x,latent_vec=False):
        x = self.encode(x)
        x_reconst = self.decode(x)
        return x_reconst

if __name__ == "__main__":

   
    # model_type = 'colormnist'
    # concept_set_type = '7'
    gpu_ids = list(np.arange(torch.cuda.device_count())) 
    parser = ArgumentParser()
    parser.add_argument("--num_imgs", default=150, type=int, help="number of imgs total")
    parser.add_argument("--pairs_vals", default="8", type=int, help="pair number")
    parser.add_argument("--model_type", default="colormnist", help="pair number")
    parser.add_argument("--dset", default="mnist", help="pair number")
    parser.add_argument("--bias", default=0, help="bias")
    parser.add_argument("--bottleneck_name", default="conv2", help="bt")



    parser.set_defaults(verbose=False)
    opt = parser.parse_args()

    model_type = opt.model_type
    dset = opt.dset
    pairs_vals = opt.pairs_vals
    bottleneck_name = opt.bottleneck_name
    bias = opt.bias
    kwargs = {'num_workers': 1, 'pin_memory': True}
    model,shape = get_vanilla_model(model_type)
    model.eval()
    model = model.cuda()

    named_layers = dict(model.named_modules())
    

    print("named_layers",named_layers.keys())
    img_data = ImgsDataset(model_type,shape,concept_set_type=str(pairs_vals),num_imgs=opt.num_imgs)
    dataloader = DataLoader(img_data, batch_size=1,
                            shuffle=True, num_workers=0)

    save_path = "/media/Data2/avani.gupta/acts_"+model_type+bottleneck_name+'/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, (img_data, con) in enumerate(dataloader):
        img_data = torch.permute(img_data.squeeze(0), (0, 3, 2, 1))
        acts = []
        for im in img_data:
            def save_activation_hook(mod, inp, out):
                global bn_activation
                bn_activation = out
            
            handle = named_layers[bottleneck_name].register_forward_hook(save_activation_hook)
            if model_type=='cg' or model_type=='iiww':
                out = model.netG.forward(im.unsqueeze(0).cuda().float())
            elif  model_type=='clever_hans' or model_type=='clever_hans7' or model_type=='cat_dog' or model_type=='faces':
                out = model(im.unsqueeze(0).cuda().float())
            elif model_type =='toy':
                im = im.view(-1,200*200)
                out = model(im.cuda().float())
            elif model_type =='toy_conv':
                out = model(im.unsqueeze(0).cuda().float())
            else: #model_type =='colormnist' or model_type=='decoymnist' or mo:
                # im = im.view(-1,200*200)
                out = model(im.unsqueeze(0).cuda().float())
            
            act = bn_activation.detach()
        
            acts.append(act)
            handle.remove()
        # breakpoint()
        acts = torch.concat(acts,axis=0)
        torch.save(acts,save_path+con[0]+'all.pt')
        print("acts",acts.shape)
        print(f"saved to {save_path+con[0]+'all.pt'}")
