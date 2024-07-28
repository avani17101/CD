import torch
import torch.nn as nn
import numpy as np
from statistics import mean
from utils.utils_tcav2 import *
import os.path
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import abc
import torchvision.models as models
import numpy as np
import abc


class ConvsUnconvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(ConvsUnconvNN,self).__init__()
        self.conv1 = nn.Conv2d(1,3,(161,66),(1,2))
        # self.conv2 = nn.ConvTranspose2d(64,3,kernel_size,stride=stride)
        # self.conv2 = nn.Conv2d(3, 3, (161,289), stride=1)

    def forward(self,x):
        x = self.conv1(x)
        return x

class ConvsDownconvNN(torch.nn.Module):
    '''
    Mainly for mapping to match dims
    '''
    def __init__(self):
        super(ConvsDownconvNN,self).__init__()
        self.conv1 = nn.ConvTranspose2d(3,1,(161,66),(1,2))

    def forward(self,x):
        x = self.conv1(x)
        return x

    
class fchead(torch.nn.Module):
    def __init__(self,input_size=196608, num_classes=2):
        super(fchead,self).__init__()
        # self.linear1 = torch.nn.Linear(input_size,589824)
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
#         self.recon_loss_type = recon_loss_type
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

        # Decoder
#         self.convTrans9 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=self.k4, stride=self.s4,
#                                padding=self.pd4),
#             nn.BatchNorm2d(512, momentum=0.01),
#             nn.ReLU(inplace=True),
#         )
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
    direcs = {}
    with open('pairs_large.pickle', 'rb') as handle:
        pairs = pickle.load(handle)
    concepts = list(pairs.keys())

    for c in pairs:
        _,r = pairs[c][0]
        concepts.append(r)

    concepts = list(set(concepts))
    print(concepts)
    upconv_m = ConvsUnconvNN()
    model = ResNet_Enco()
    downconv_m = ConvsDownconvNN()
    mse_loss = nn.MSELoss()
    model.train()
    upconv_m.train()
    downconv_m.train()
    model = model.cuda()
    upconv_m = upconv_m.cuda()
    downconv_m = downconv_m.cuda()

    learning_rate = 0.001
    import random
    num_imgs = 44
    model_params = list(model.parameters())+list(upconv_m.parameters())+list(downconv_m.parameters())
    optimizer = torch.optim.SGD([p for p in model_params if p.requires_grad],lr=learning_rate)
    loss_lis = []
    path_nps = '/media/Data2/avani.gupta/imgs_np/'
    checkpoint_path='CGIntrinsics/checkpoints/test_local/cgintrinsics_iiw_saw_final_net_G.pth'
    opt = TrainOptions().parse() 

    CGmodel = create_model(opt,checkpoint_path)
    CGmodel.netG.eval()
    CGmodel.netG = CGmodel.netG.cuda()
    for p in CGmodel.netG.parameters():
        p.requires_grad = False

    for branch in ['S']:
        for ep in tqdm(range(50)):
            
            for con in tqdm(concepts):
                imgs = np.load(path_nps+con+'.npy')
                imgs = np.moveaxis(imgs,3,1)
                imgs = torch.from_numpy(imgs)
                imgs = imgs.cuda()
                out = CGmodel.netG(imgs)
                R, S = out
                if branch=='R':
                    acts = R
                else:
                    acts = S
                x = upconv_m(acts)
                x_e = model.encode(x)
                x_d = model.decode(x_e)
                x_fea = downconv_m(x_d)
                loss = mse_loss(x_d, x) + mse_loss(x_fea, acts)

                loss.backward()
                assert(torch.isnan(torch.tensor(loss.item()))==False)
                loss_lis.append(loss.item())

                optimizer.step()
                optimizer.zero_grad()
            print("ep",ep," loss",mean(loss_lis))
            path = '/media/Data2/avani.gupta/student/'+'branch'+branch+'/'
            if not os.path.exists(path):
                os.makedirs(path)

            torch.save(model.state_dict(), path+'encodeco.pt')
            torch.save(upconv_m.state_dict(), path+'upconv.pt')
            torch.save(downconv_m.state_dict(), path+'downconv.pt')
